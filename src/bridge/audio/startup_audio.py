"""Bounded PCM retained while the provider handshake runs; no worker or network."""
from collections import deque

class StartupAudio:
    def __init__(self, frames):
        if frames < 1: raise ValueError('startup_audio_frames must be positive')
        self.limit = frames
        self.pending = deque()
        self.dropped = 0

    def admit(self, packet):
        if len(self.pending) >= self.limit:
            self.dropped += 1
            return False
        self.pending.append(packet)
        return True

    def clear(self):
        self.pending.clear(); self.dropped=0
