import wave
from datetime import datetime
from pathlib import Path
from typing import Optional


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
