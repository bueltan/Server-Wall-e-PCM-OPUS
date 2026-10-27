import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class BridgeConfig:
    host: str = "0.0.0.0"
    port: int = 9999

# Selection applies to continuous UDP v3; legacy adapters remain xAI-only.
    voice_provider: str = field(default_factory=lambda: os.getenv("PIUU_VOICE_PROVIDER", "gemini").strip().lower())
    gemini_api_key: Optional[str] = field(default_factory=lambda: os.getenv("GEMINI_API_KEY"))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-3.8-live"))
    gemini_voice: str = field(default_factory=lambda: os.getenv("GEMINI_VOICE", "voice_nvriqqd72uh5"))
    gemini_prompt: Optional[str] = None  # Reuse the PIUU persona unless overridden.
    gemini_realtime_url: str = (
        "wss://generativelanguage.googleapis.com/ws/"
        "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
    )

    xai_realtime_url: str = "wss://api.x.ai/v1/realtime?model=grok-voice-latest"  # ← actualizado
    xai_api_key: Optional[str] = os.getenv("XAI_API_KEY")
    xai_voice: str = "eve"
    xai_prompt: str = (
        "Sos PIUU, un pajarito amable, simpático y curioso. "
        "Hablá en español, con calidez, naturalidad y un poco de juego. "
        "Respondé de forma breve y útil. Podés cerrar con una pregunta suave cuando "
        "ayude a la conversación; no fuerces preguntas ni prolongues despedidas "
        "o confirmaciones. Adaptá el tono al contexto sin exagerar emociones."
    )
    # Continuous v3 provider tool and optional capability-negotiated visual transport.
    expression_tools_enabled: bool = True
    expression_udp_enabled: bool = True


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
    opus_target_bitrate: int = 20000
    opus_encoder_complexity: int = 8
    opus_expected_packet_loss_perc: int = 3
    opus_use_vbr: int = 1
    opus_use_inband_fec: int = 1

    # Bounded per-client uplink. Audio drops never consume END/COMMIT slots.
    input_queue_frames: int = 16
    input_reorder_window_sec: float = 0.04
    input_recovery_window_sec: float = 0.40
    input_reorder_frames: int = 32  # separate from WS admission; 640 ms of PCM
    input_silence_max_ms: int = 200
    input_send_timeout_sec: float = 1.0

    # Ordered PCM retained while history repair blocks the provider, not UDP.
    provider_input_buffer_sec: float = 5.0
    provider_input_max_age_sec: float = 5.0
    provider_input_send_interval_sec: float = .01  # at most 2x real time to catch up

    # Independent deadlines: CORE liveness and provider history transactions.
    core_disconnect_timeout_sec: float = 10.0
    history_repair_timeout_sec: float = 4.0
    history_item_timeout_sec: float = 1.0
    history_confirmation_timeout_sec: float = 1.0

    # Provider is opened only after START; bound audio retained during handshake.
    provider_open_timeout_sec: float = 6.0
    startup_audio_frames: int = 300  # six seconds, at most 197400 raw bytes

    # UDP output pacing toward ESP32
    output_send_interval_sec: float = 0.02
    output_pacing_max_lag_sec: float = 0.04  # rebase after an overrun, no unlimited catch-up
    output_end_repeat_count: int = 5
    output_end_repeat_interval: float = 0.02

    # Bounded downlink recovery: cache is on the server, CORE keeps its jitter buffer.
    output_cache_frames: int = 128
    output_cache_ttl_sec: float = 3.0
    output_end_retry_sec: float = 0.15
    output_end_timeout_sec: float = 5.0

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
