import struct
from .config import BridgeConfig


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
