"""Stateful anti-aliasing conversion from Gemini PCM24k to CORE PCM16k."""
import numpy as np
from scipy.signal import firwin, lfilter


class Pcm24To16:
    # Rational conversion at 48 kHz: interpolate x2, low-pass, decimate /3.
    # State and decimation phase survive arbitrary network chunk boundaries.
    taps = firwin(121, 1 / 3, window=('kaiser', 7.0)) * 2

    def __init__(self):
        self.state = np.zeros(len(self.taps) - 1)
        self.phase = 0
        self.used = False

    def _filter(self, samples):
        filtered, self.state = lfilter(self.taps, [1.0], samples, zi=self.state)
        output = filtered[(-self.phase) % 3::3]
        self.phase = (self.phase + len(samples)) % 3
        return np.clip(np.rint(output), -32768, 32767).astype('<i2').tobytes()

    def convert(self, pcm):
        if len(pcm) % 2:
            raise ValueError('Gemini PCM must contain complete PCM16 samples')
        if not pcm:
            return b''
        self.used = True
        samples = np.frombuffer(pcm, dtype='<i2')
        upsampled = np.zeros(len(samples) * 2)
        upsampled[::2] = samples
        return self._filter(upsampled)

    def finish(self):
        if not self.used:
            return b''
        self.used = False
        return self._filter(np.zeros(len(self.taps) - 1))
