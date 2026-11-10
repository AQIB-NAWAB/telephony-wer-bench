"""Audio helpers: file IO, resampling, framing and a small energy VAD."""

from __future__ import annotations

from collections.abc import Iterator
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

FRAME_MS = 20


def read_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Read an audio file as mono float32 in [-1, 1]."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return data.mean(axis=1).astype(np.float32), int(sr)


def write_wav(path: str | Path, audio: np.ndarray, sr: int) -> None:
    """Write mono 16-bit PCM WAV."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), to_int16(audio), sr, subtype="PCM_16")


def to_int16(audio: np.ndarray) -> np.ndarray:
    return np.clip(np.round(audio * 32768.0), -32768, 32767).astype(np.int16)


def to_float(pcm: np.ndarray) -> np.ndarray:
    return pcm.astype(np.float32) / 32768.0


def resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Polyphase resample with the default Kaiser anti-alias filter."""
    if sr_in == sr_out:
        return audio.astype(np.float32, copy=False)
    g = gcd(sr_in, sr_out)
    out = resample_poly(audio, sr_out // g, sr_in // g)
    return out.astype(np.float32)


def frame_samples(sr: int, frame_ms: int = FRAME_MS) -> int:
    return sr * frame_ms // 1000


def iter_frames(audio: np.ndarray, sr: int, frame_ms: int = FRAME_MS) -> Iterator[np.ndarray]:
    """Yield fixed-size int16 frames. The last frame is zero padded."""
    n = frame_samples(sr, frame_ms)
    pcm = to_int16(audio)
    for i in range(0, len(pcm), n):
        chunk = pcm[i : i + n]
        if len(chunk) < n:
            chunk = np.pad(chunk, (0, n - len(chunk)))
        yield chunk


def rms(x: np.ndarray) -> float:
    if len(x) == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


def speech_bounds(audio: np.ndarray, sr: int, frame_ms: int = 10, margin_db: float = 30.0) -> tuple[float, float]:
    """Find first and last active frame using a simple energy threshold.

    A frame is active when its RMS is within ``margin_db`` of the loudest 10 ms
    frame (or above a small absolute floor). Returns (start_s, end_s). It is
    only used to mark where speech begins and ends so latency can be measured.
    """
    n = frame_samples(sr, frame_ms)
    if len(audio) < n:
        return 0.0, len(audio) / sr
    count = len(audio) // n
    frames = audio[: count * n].reshape(count, n).astype(np.float64)
    level = np.sqrt((frames**2).mean(axis=1))
    peak = float(level.max())
    if peak <= 1e-6:
        return 0.0, len(audio) / sr
    threshold = max(peak * 10 ** (-margin_db / 20.0), 1e-4)
    active = np.flatnonzero(level >= threshold)
    return float(active[0] * frame_ms / 1000), float((active[-1] + 1) * frame_ms / 1000)
