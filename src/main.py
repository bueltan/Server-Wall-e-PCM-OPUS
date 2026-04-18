import asyncio
import base64
import ctypes
import ctypes.util
import json
import os
import struct
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import websockets
from websockets.exceptions import ConnectionClosed


# ============================================================
# Exceptions
# ============================================================

class OpusError(RuntimeError):
    pass


# ============================================================
# Config
# ============================================================

@dataclass(frozen=True)
class BridgeConfig:
    host: str = "0.0.0.0"
    port: int = 9999

    xai_realtime_url: str = "wss://api.x.ai/v1/realtime"
    xai_api_key: Optional[str] = os.getenv("XAI_API_KEY")
    xai_voice: str = "eve"
    xai_prompt: str = (
        "You are WALL-E, a warm and helpful voice assistant. "
        "Be concise, friendly, and natural."
    )

    input_sample_rate: int = 16000
    output_sample_rate: int = 16000
    channels: int = 1
    sample_width: int = 2  # PCM16 mono

    # Uplink packet protocol from ESP32 -> Python
    pcm_magic: bytes = b"PCM!"

    # Downlink packet protocol from Python -> ESP32
    opus_magic: bytes = b"OPUS"
    version: int = 1
    flag_end: int = 0x01

    # Opus encoder settings for downlink
    opus_application_voip: int = 2048
    opus_ok: int = 0
    opus_set_bitrate_request: int = 4002
    opus_set_complexity_request: int = 4010
    opus_set_vbr_request: int = 4006
    opus_set_signal_request: int = 4024
    opus_set_packet_loss_perc_request: int = 4014
    opus_set_inband_fec_request: int = 4012
    opus_signal_voice: int = 3001

    opus_frame_ms: int = 20
    opus_target_bitrate: int = 12000
    opus_encoder_complexity: int = 5
    opus_expected_packet_loss_perc: int = 8
    opus_use_vbr: int = 0
    opus_use_inband_fec: int = 1

    # UDP output pacing toward ESP32
    output_send_interval_sec: float = 0.02
    output_end_repeat_count: int = 5
    output_end_repeat_interval: float = 0.02

    # Debug recordings
    save_input_wav: bool = True
    save_output_wav: bool = True
    recordings_dir: str = "recordings"

    @property
    def opus_frame_samples(self) -> int:
        return self.output_sample_rate * self.opus_frame_ms // 1000

    @property
    def opus_frame_bytes(self) -> int:
        return self.opus_frame_samples * self.channels * self.sample_width


# ============================================================
# Packet formats
# ============================================================

class BasePacketFormat:
    HEADER_STRUCT = struct.Struct("!4sBBIIHH")

    def __init__(self, magic: bytes, version: int, flag_end: int):
        self.magic = magic
        self.version = version
        self.flag_end = flag_end

    def pack_audio_packet(
        self,
        seq: int,
        pts_samples: int,
        frame_samples: int,
        payload: bytes,
    ) -> bytes:
        header = self.HEADER_STRUCT.pack(
            self.magic,
            self.version,
            0,
            seq,
            pts_samples,
            frame_samples,
            len(payload),
        )
        return header + payload

    def pack_end_packet(self, seq: int, pts_samples: int) -> bytes:
        return self.HEADER_STRUCT.pack(
            self.magic,
            self.version,
            self.flag_end,
            seq,
            pts_samples,
            0,
            0,
        )

    def unpack_packet(self, data: bytes) -> dict:
        if len(data) < self.HEADER_STRUCT.size:
            raise ValueError(
                f"Packet too short: got {len(data)}, expected at least {self.HEADER_STRUCT.size}"
            )

        magic, version, flags, seq, pts_samples, frame_samples, payload_len = (
            self.HEADER_STRUCT.unpack(data[: self.HEADER_STRUCT.size])
        )
        payload = data[self.HEADER_STRUCT.size :]

        return {
            "magic": magic,
            "version": version,
            "flags": flags,
            "seq": seq,
            "pts_samples": pts_samples,
            "frame_samples": frame_samples,
            "payload_len": payload_len,
            "payload": payload,
        }

    def validate_packet(self, packet: dict) -> None:
        if packet["magic"] != self.magic:
            raise ValueError(f"Invalid magic: {packet['magic']!r}")
        if packet["version"] != self.version:
            raise ValueError(f"Invalid version: {packet['version']}")

        expected_total = self.HEADER_STRUCT.size + packet["payload_len"]
        actual_total = self.HEADER_STRUCT.size + len(packet["payload"])
        if expected_total != actual_total:
            raise ValueError(
                f"Packet size mismatch: expected={expected_total}, actual={actual_total}"
            )

    def is_end_packet(self, packet: dict) -> bool:
        return (packet["flags"] & self.flag_end) != 0


class PcmPacketFormat(BasePacketFormat):
    def __init__(self, config: BridgeConfig):
        super().__init__(config.pcm_magic, config.version, config.flag_end)


class OpusPacketFormat(BasePacketFormat):
    def __init__(self, config: BridgeConfig):
        super().__init__(config.opus_magic, config.version, config.flag_end)


# ============================================================
# WAV recorder
# ============================================================

class RollingWavRecorder:
    def __init__(self, sample_rate: int, channels: int, sample_width: int, base_dir: Path):
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.wave_file: Optional[wave.Wave_write] = None
        self.path: Optional[Path] = None

    def start(self, prefix: str, suffix: str = "") -> None:
        self.close()
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.path = self.base_dir / f"{prefix}_{stamp}{suffix}.wav"
        self.wave_file = wave.open(str(self.path), "wb")
        self.wave_file.setnchannels(self.channels)
        self.wave_file.setsampwidth(self.sample_width)
        self.wave_file.setframerate(self.sample_rate)
        print(f"[WAV] recording -> {self.path}")

    def write(self, pcm_bytes: bytes) -> None:
        if self.wave_file is not None:
            self.wave_file.writeframes(pcm_bytes)

    def close(self) -> None:
        if self.wave_file is not None:
            try:
                self.wave_file.close()
            finally:
                print(f"[WAV] closed -> {self.path}")
                self.wave_file = None
                self.path = None


# ============================================================
# libopus encoder for downlink
# ============================================================

class LibOpusBase:
    def __init__(self, config: BridgeConfig):
        self.config = config
        self.lib = self._load_libopus()

    def _load_libopus(self):
        candidates = []
        found = ctypes.util.find_library("opus")
        if found:
            candidates.append(found)

        candidates.extend(
            [
                "libopus.so",
                "libopus.so.0",
                "libopus.dylib",
                "opus.dll",
                "libopus-0.dll",
            ]
        )

        last_error = None
        for name in candidates:
            try:
                return ctypes.CDLL(name)
            except OSError as exc:
                last_error = exc

        raise OpusError(f"Could not load libopus: {last_error}")


class LibOpusEncoder(LibOpusBase):
    def __init__(self, config: BridgeConfig):
        super().__init__(config)
        self.encoder = None
        self._configure_signatures()
        self._create_encoder()
        self._configure_encoder()

    def _configure_signatures(self):
        self.lib.opus_encoder_create.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
        ]
        self.lib.opus_encoder_create.restype = ctypes.c_void_p

        self.lib.opus_encode.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int16),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_int,
        ]
        self.lib.opus_encode.restype = ctypes.c_int

        self.lib.opus_encoder_destroy.argtypes = [ctypes.c_void_p]
        self.lib.opus_encoder_destroy.restype = None

        self.lib.opus_encoder_ctl.restype = ctypes.c_int

    def _create_encoder(self):
        err = ctypes.c_int(0)
        self.encoder = self.lib.opus_encoder_create(
            self.config.output_sample_rate,
            self.config.channels,
            self.config.opus_application_voip,
            ctypes.byref(err),
        )
        if not self.encoder or err.value != self.config.opus_ok:
            raise OpusError(f"opus_encoder_create failed: {err.value}")

    def _ctl(self, request: int, value: int):
        result = self.lib.opus_encoder_ctl(self.encoder, request, value)
        if result != self.config.opus_ok:
            raise OpusError(f"opus_encoder_ctl({request}, {value}) failed: {result}")

    def _configure_encoder(self):
        self._ctl(self.config.opus_set_bitrate_request, self.config.opus_target_bitrate)
        self._ctl(self.config.opus_set_complexity_request, self.config.opus_encoder_complexity)
        self._ctl(self.config.opus_set_vbr_request, self.config.opus_use_vbr)
        self._ctl(self.config.opus_set_signal_request, self.config.opus_signal_voice)
        self._ctl(
            self.config.opus_set_packet_loss_perc_request,
            self.config.opus_expected_packet_loss_perc,
        )
        self._ctl(
            self.config.opus_set_inband_fec_request,
            self.config.opus_use_inband_fec,
        )

    def encode_pcm16_frame(self, pcm16_frame: bytes) -> bytes:
        expected_len = self.config.opus_frame_bytes
        if len(pcm16_frame) != expected_len:
            raise ValueError(
                f"Invalid PCM frame size: got {len(pcm16_frame)}, expected {expected_len}"
            )

        pcm_array = (ctypes.c_int16 * self.config.opus_frame_samples).from_buffer_copy(
            pcm16_frame
        )
        max_packet_bytes = 512
        out_buf = (ctypes.c_ubyte * max_packet_bytes)()

        encoded_len = self.lib.opus_encode(
            self.encoder,
            pcm_array,
            self.config.opus_frame_samples,
            out_buf,
            max_packet_bytes,
        )
        if encoded_len < 0:
            raise OpusError(f"opus_encode failed: {encoded_len}")

        return bytes(out_buf[:encoded_len])

    def close(self):
        if self.encoder:
            self.lib.opus_encoder_destroy(self.encoder)
            self.encoder = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ============================================================
# xAI realtime bridge
# ============================================================

class XaiRealtimeBridge:
    def __init__(self, config: BridgeConfig, output_pcm_queue: asyncio.Queue[bytes | None]):
        self.config = config
        self.output_pcm_queue = output_pcm_queue

        self.ws = None
        self.connected_event = asyncio.Event()
        self.send_lock = asyncio.Lock()

        self.output_audio_bytes = 0
        self.output_audio_chunks = 0

    async def connect(self) -> None:
        api_key = self.config.xai_api_key or os.getenv("XAI_API_KEY")
        if not api_key:
            raise RuntimeError("XAI_API_KEY is not set")

        print("[XAI] connecting...")
        self.ws = await websockets.connect(
            self.config.xai_realtime_url,
            additional_headers={"Authorization": f"Bearer {api_key}"},
            open_timeout=20,
            close_timeout=10,
            ping_interval=15,
            ping_timeout=30,
            max_queue=64,
        )
        self.connected_event.set()
        await self._send_session_update()
        print("[XAI] connected")

    async def _safe_send(self, payload: dict) -> None:
        if self.ws is None or not self.connected_event.is_set():
            raise RuntimeError("xAI websocket is not connected")
        async with self.send_lock:
            await self.ws.send(json.dumps(payload))

    async def _send_session_update(self) -> None:
        await self._safe_send(
            {
                "type": "session.update",
                "session": {
                    "voice": self.config.xai_voice,
                    "instructions": self.config.xai_prompt,
                    "turn_detection": None,
                    "input_audio_transcription": {"model": "grok-2-audio"},
                    "audio": {
                        "input": {
                            "format": {
                                "type": "audio/pcm",
                                "rate": self.config.input_sample_rate,
                            }
                        },
                        "output": {
                            "format": {
                                "type": "audio/pcm",
                                "rate": self.config.output_sample_rate,
                            }
                        },
                    },
                },
            }
        )

    async def send_audio_append(self, pcm_data: bytes) -> bool:
        if not self.connected_event.is_set():
            print("[XAI] append skipped: not connected")
            return False

        try:
            await self._safe_send(
                {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm_data).decode("utf-8"),
                }
            )
            return True
        except Exception as exc:
            print(f"[XAI] append failed: {exc}")
            return False

    async def send_commit(self) -> bool:
        if not self.connected_event.is_set():
            print("[XAI] commit skipped: not connected")
            return False

        self.output_audio_bytes = 0
        self.output_audio_chunks = 0

        try:
            await self._safe_send({"type": "input_audio_buffer.commit"})
            await self._safe_send({"type": "response.create"})
            print("[XAI] commit + response.create sent")
            return True
        except Exception as exc:
            print(f"[XAI] commit failed: {exc}")
            return False

    async def receive_events_forever(self) -> None:
        if self.ws is None:
            raise RuntimeError("receive_events_forever called before connect")

        async for raw_event in self.ws:
            event = json.loads(raw_event)
            event_type = event.get("type")

            if event_type == "session.created":
                session_id = event.get("session", {}).get("id")
                print(f"[XAI] session.created id={session_id}")

            elif event_type == "session.updated":
                print("[XAI] session.updated")

            elif event_type == "input_audio_buffer.committed":
                print("[XAI] input_audio_buffer.committed")

            elif event_type == "conversation.item.input_audio_transcription.completed":
                transcript = event.get("transcript", "").strip()
                if transcript:
                    print(f"[USER] {transcript}")

            elif event_type == "response.output_audio.delta":
                pcm_data = base64.b64decode(event["delta"])
                self.output_audio_chunks += 1
                self.output_audio_bytes += len(pcm_data)

                if self.output_audio_chunks <= 5 or self.output_audio_chunks % 25 == 0:
                    samples = len(pcm_data) // 2
                    approx_ms = (samples / self.config.output_sample_rate) * 1000.0
                    print(
                        "[XAI] output_audio.delta "
                        f"chunk={self.output_audio_chunks} bytes={len(pcm_data)} ~{approx_ms:.1f}ms"
                    )

                await self.output_pcm_queue.put(pcm_data)

            elif event_type == "response.output_audio_transcript.delta":
                delta_text = event.get("delta", "")
                if delta_text:
                    print(delta_text, end="", flush=True)

            elif event_type == "response.output_audio_transcript.done":
                transcript = event.get("transcript", "").strip()
                if transcript:
                    print(f"\n[ASSISTANT] {transcript}")

            elif event_type == "response.output_audio.done":
                total_samples = self.output_audio_bytes // 2
                total_ms = (total_samples / self.config.output_sample_rate) * 1000.0
                print(
                    "[XAI] response.output_audio.done "
                    f"chunks={self.output_audio_chunks} bytes={self.output_audio_bytes} ~{total_ms:.1f}ms"
                )
                await self.output_pcm_queue.put(None)

            elif event_type == "response.done":
                usage = event.get("usage", {})
                print(f"[XAI] response.done tokens={usage.get('total_tokens')}")

            elif event_type == "ping":
                pass

            elif event_type == "error":
                print(f"[XAI] error: {json.dumps(event, ensure_ascii=False)}")

            else:
                print(f"[XAI] unhandled event: {event_type}")

    async def close(self) -> None:
        self.connected_event.clear()
        if self.ws is not None:
            try:
                await self.ws.close()
            except Exception:
                pass
            self.ws = None


# ============================================================
# UDP protocol bridge
# ============================================================

class UdpAudioBridgeProtocol(asyncio.DatagramProtocol):
    def __init__(self, config: BridgeConfig):
        self.config = config
        self.transport = None

        self.pcm_packet_format = PcmPacketFormat(config)
        self.opus_packet_format = OpusPacketFormat(config)
        self.opus_encoder = LibOpusEncoder(config)

        self.output_pcm_queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self.xai = XaiRealtimeBridge(config, self.output_pcm_queue)

        self.remote_addr: Optional[Tuple[str, int]] = None
        self.inbound_started = False
        self.output_active = False

        recordings_dir = Path(config.recordings_dir)
        self.input_recorder = RollingWavRecorder(
            sample_rate=config.input_sample_rate,
            channels=config.channels,
            sample_width=config.sample_width,
            base_dir=recordings_dir,
        )
        self.output_recorder = RollingWavRecorder(
            sample_rate=config.output_sample_rate,
            channels=config.channels,
            sample_width=config.sample_width,
            base_dir=recordings_dir,
        )

        self.output_task: Optional[asyncio.Task] = None
        self.xai_receive_task: Optional[asyncio.Task] = None

        self.output_sequence = 0
        self.output_pts_samples = 0
        self.output_pending_pcm = bytearray()

    def connection_made(self, transport) -> None:
        self.transport = transport
        print("[BRIDGE] UDP bridge ready")
        print(f"[BRIDGE] listening on udp://{self.config.host}:{self.config.port}")
        print("[BRIDGE] uplink: PCM UDP -> xAI PCM websocket")
        print("[BRIDGE] downlink: xAI PCM websocket -> Opus UDP")

        loop = asyncio.get_running_loop()
        self.xai_receive_task = loop.create_task(self._run_xai())
        self.output_task = loop.create_task(self._udp_output_sender())

    async def _run_xai(self) -> None:
        while True:
            try:
                await self.xai.connect()
                await self.xai.receive_events_forever()
            except ConnectionClosed as exc:
                print(
                    f"[XAI] websocket closed code={getattr(exc, 'code', None)} "
                    f"reason={getattr(exc, 'reason', None)}"
                )
            except Exception as exc:
                print(f"[XAI] connection loop error: {exc}")
            finally:
                self.xai.connected_event.clear()
                await asyncio.sleep(2.0)

    def datagram_received(self, data: bytes, addr: Tuple[str, int]) -> None:
        if data == b"PING":
            self.remote_addr = addr
            self.transport.sendto(b"PONG", addr)
            return

        if data == b"COMMIT":
            self.remote_addr = addr
            asyncio.create_task(self._handle_commit(addr))
            return

        if len(data) >= 4 and data[:4] == self.config.pcm_magic:
            self.remote_addr = addr
            asyncio.create_task(self._handle_pcm_packet(data, addr))
            return

        if len(data) >= 4 and data[:4] == self.config.opus_magic:
            print(f"[BRIDGE] unexpected inbound OPUS packet from {addr}, ignoring")
            return

        try:
            msg = data.decode(errors="ignore").strip()
        except Exception:
            msg = "<binary>"
        print(f"[BRIDGE] unknown datagram from {addr}: {msg[:80]}")

    async def _handle_commit(self, addr: Tuple[str, int]) -> None:
        print(f"[BRIDGE] COMMIT from {addr}")
        self._reset_downlink_turn_state()
        ok = await self.xai.send_commit()
        if not ok:
            print("[BRIDGE] COMMIT failed")

    def _reset_downlink_turn_state(self) -> None:
        self.output_sequence = 0
        self.output_pts_samples = 0
        self.output_pending_pcm.clear()
        self.output_active = False
        self.output_recorder.close()

        drained = 0
        while True:
            try:
                self.output_pcm_queue.get_nowait()
                drained += 1
            except asyncio.QueueEmpty:
                break

        if drained > 0:
            print(f"[BRIDGE] cleared {drained} stale downlink queue item(s)")

    async def _handle_pcm_packet(self, data: bytes, addr: Tuple[str, int]) -> None:
        try:
            packet = self.pcm_packet_format.unpack_packet(data)
            self.pcm_packet_format.validate_packet(packet)

            if self.pcm_packet_format.is_end_packet(packet):
                print(
                    f"[BRIDGE] input END from {addr} seq={packet['seq']} pts={packet['pts_samples']}"
                )
                self.inbound_started = False
                self.input_recorder.close()
                return

            payload = packet["payload"]
            frame_samples = packet["frame_samples"]
            expected_payload_len = frame_samples * self.config.sample_width * self.config.channels
            if packet["payload_len"] != expected_payload_len:
                raise ValueError(
                    f"Invalid payload len: got={packet['payload_len']} expected={expected_payload_len}"
                )

            if not self.inbound_started:
                self.inbound_started = True
                if self.config.save_input_wav:
                    suffix = f"_{addr[0].replace('.', '_')}_{addr[1]}"
                    self.input_recorder.start("uplink", suffix=suffix)

            if self.config.save_input_wav:
                self.input_recorder.write(payload)

            ok = await self.xai.send_audio_append(payload)
            if not ok:
                print("[BRIDGE] dropped uplink frame because xAI append failed")
                return

            seq = packet["seq"]
            if seq < 8 or seq % 50 == 0:
                print(
                    f"[BRIDGE] uplink seq={seq} pts={packet['pts_samples']} "
                    f"samples={frame_samples} bytes={len(payload)}"
                )

        except Exception as exc:
            print(f"[BRIDGE] input packet error from {addr}: {exc}")

    async def _udp_output_sender(self) -> None:
        while True:
            chunk = await self.output_pcm_queue.get()

            if chunk is None:
                await self._flush_pending_pcm(final_flush=True)
                await self._send_output_end_packets()
                continue

            if self.remote_addr is None:
                continue

            if not self.output_active:
                self.output_active = True
                if self.config.save_output_wav:
                    suffix = f"_{self.remote_addr[0].replace('.', '_')}_{self.remote_addr[1]}"
                    self.output_recorder.start("downlink", suffix=suffix)

            if self.config.save_output_wav:
                self.output_recorder.write(chunk)

            self.output_pending_pcm.extend(chunk)
            await self._flush_pending_pcm(final_flush=False)

    async def _flush_pending_pcm(self, final_flush: bool) -> None:
        if self.remote_addr is None:
            self.output_pending_pcm.clear()
            return

        while len(self.output_pending_pcm) >= self.config.opus_frame_bytes:
            pcm_frame = bytes(self.output_pending_pcm[: self.config.opus_frame_bytes])
            del self.output_pending_pcm[: self.config.opus_frame_bytes]
            self._send_output_opus_frame(pcm_frame)
            await asyncio.sleep(self.config.output_send_interval_sec)

        if final_flush and self.output_pending_pcm:
            pcm_frame = bytes(self.output_pending_pcm)
            self.output_pending_pcm.clear()
            if len(pcm_frame) < self.config.opus_frame_bytes:
                pcm_frame += b"\x00" * (self.config.opus_frame_bytes - len(pcm_frame))
            self._send_output_opus_frame(pcm_frame)
            await asyncio.sleep(self.config.output_send_interval_sec)

    def _send_output_opus_frame(self, pcm_frame: bytes) -> None:
        if self.transport is None or self.remote_addr is None:
            return

        opus_payload = self.opus_encoder.encode_pcm16_frame(pcm_frame)
        packet = self.opus_packet_format.pack_audio_packet(
            seq=self.output_sequence,
            pts_samples=self.output_pts_samples,
            frame_samples=self.config.opus_frame_samples,
            payload=opus_payload,
        )
        self.transport.sendto(packet, self.remote_addr)

        if self.output_sequence < 8 or self.output_sequence % 50 == 0:
            print(
                f"[BRIDGE] downlink OPUS seq={self.output_sequence} "
                f"pts={self.output_pts_samples} opus_bytes={len(opus_payload)} udp_bytes={len(packet)}"
            )

        self.output_sequence += 1
        self.output_pts_samples += self.config.opus_frame_samples

    async def _send_output_end_packets(self) -> None:
        if self.transport is None or self.remote_addr is None or not self.output_active:
            return

        for i in range(self.config.output_end_repeat_count):
            end_packet = self.opus_packet_format.pack_end_packet(
                seq=self.output_sequence,
                pts_samples=self.output_pts_samples,
            )
            self.transport.sendto(end_packet, self.remote_addr)
            print(
                f"[BRIDGE] sent OPUS END {i + 1}/{self.config.output_end_repeat_count} "
                f"seq={self.output_sequence} pts={self.output_pts_samples}"
            )
            if i + 1 < self.config.output_end_repeat_count:
                await asyncio.sleep(self.config.output_end_repeat_interval)

        self.output_active = False
        self.output_recorder.close()

    def connection_lost(self, exc) -> None:
        print(f"[BRIDGE] UDP connection lost: {exc}")
        self.input_recorder.close()
        self.output_recorder.close()
        self.opus_encoder.close()

        if self.output_task is not None:
            self.output_task.cancel()
        if self.xai_receive_task is not None:
            self.xai_receive_task.cancel()


# ============================================================
# App
# ============================================================

async def async_main() -> None:
    config = BridgeConfig()
    loop = asyncio.get_running_loop()

    transport, _protocol = await loop.create_datagram_endpoint(
        lambda: UdpAudioBridgeProtocol(config),
        local_addr=(config.host, config.port),
    )

    try:
        await asyncio.Future()
    finally:
        transport.close()


if __name__ == "__main__":
    asyncio.run(async_main())
