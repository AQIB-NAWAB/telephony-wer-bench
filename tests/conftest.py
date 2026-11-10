import numpy as np
import pytest


@pytest.fixture
def speechish():
    """One second of a modulated tone at 16 kHz, loud enough to look like speech to an energy VAD."""
    sr = 16000
    t = np.arange(sr) / sr
    x = 0.3 * np.sin(2 * np.pi * 220 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))
    return np.concatenate([np.zeros(sr // 4), x, np.zeros(sr // 4)]).astype(np.float32), sr
