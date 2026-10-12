import ctypes
import ctypes.util
from bridge.config import BridgeConfig


class OpusError(RuntimeError):
    pass


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
