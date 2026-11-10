"""Stateful 8 kHz to 16 kHz upsampling for frame-by-frame streaming."""

from __future__ import annotations

import numpy as np
from scipy.signal import firwin, lfilter


class Upsampler2x:
    """Zero-stuff by two and low-pass, carrying filter state across frames.

    Used to feed 8 kHz telephony audio to models trained at 16 kHz without
    edge artifacts at every 20 ms frame boundary. Adds a fixed delay of
    (taps - 1) / 2 samples at 16 kHz (about 2 ms with the default 65 taps).
    """

    def __init__(self, taps: int = 65) -> None:
        self.b = firwin(taps, 0.5, window=("kaiser", 7.0)) * 2.0
        self.zi = np.zeros(len(self.b) - 1)

    def process(self, frame: np.ndarray) -> np.ndarray:
        x = np.zeros(len(frame) * 2, dtype=np.float64)
        x[::2] = frame.astype(np.float64)
        y, self.zi = lfilter(self.b, [1.0], x, zi=self.zi)
        return np.clip(np.round(y), -32768, 32767).astype(np.int16)


def to_model_rate(sample_rate: int, model_rate: int = 16000):
    """Return a function mapping a frame at ``sample_rate`` to ``model_rate``."""
    if sample_rate == model_rate:
        return lambda f: f
    if sample_rate * 2 == model_rate:
        return Upsampler2x().process
    raise ValueError(f"cannot convert {sample_rate} Hz to {model_rate} Hz")
