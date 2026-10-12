import asyncio
from pathlib import Path
from typing import Optional, Tuple
from websockets.exceptions import ConnectionClosed
from bridge.config import BridgeConfig
from bridge.transport.protocols.audio import PcmPacketFormat, OpusPacketFormat
from bridge.audio.codec import LibOpusEncoder
from bridge.audio.recording import RollingWavRecorder
from bridge.providers.xai.realtime import XaiRealtimeBridge
from bridge.transport.ingress import InboundSession


class ClientSession:
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

        self.input = InboundSession(self.pcm_packet_format, config,
            self._accept_audio, self._accept_end, self._commit_input)

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
        self.input.start()
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
        # PING is transport reachability, not ownership or xAI readiness.
        if data == b"PING":
            self.transport.sendto(b"PONG", addr)
            return
        if self.remote_addr is not None and addr != self.remote_addr:
            return
        if data == b"COMMIT":
            if self.remote_addr is not None:
                self.input.admit_commit()
            return
        if data[:4] == self.config.pcm_magic:
            # Bind only a structurally valid AUDIO packet, never PING or END.
            if self.remote_addr is None:
                try:
                    packet = self.input.validate(data)
                except ValueError:
                    self.input.stats.invalid += 1
                    return
                if self.pcm_packet_format.is_end_packet(packet):
                    return
                self.remote_addr = addr
            self.input.admit(data)

    async def _accept_audio(self, packet: dict) -> bool:
        if not self.inbound_started:
            self.inbound_started = True
            if self.config.save_input_wav:
                suffix = f"_{self.remote_addr[0].replace('.', '_')}_{self.remote_addr[1]}"
                self.input_recorder.start("uplink", suffix=suffix)
        # Record the accepted local stream, including a frame whose WS send fails.
        if self.config.save_input_wav:
            self.input_recorder.write(packet["payload"])
        return await self.xai.send_audio_append(packet["payload"])

    async def _accept_end(self, packet: dict) -> None:
        self.inbound_started = False
        self.input_recorder.close()

    async def _commit_input(self) -> bool:
        # Also finalize recordings if the UDP END packet was lost.
        self.inbound_started = False
        self.input_recorder.close()
        self._reset_downlink_turn_state()
        return await self.xai.send_commit()

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
        self.input.close()
        self.input_recorder.close()
        self.output_recorder.close()
        self.opus_encoder.close()

        if self.output_task is not None:
            self.output_task.cancel()
        if self.xai_receive_task is not None:
            self.xai_receive_task.cancel()
