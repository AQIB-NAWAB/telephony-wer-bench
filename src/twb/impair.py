"""Network and acoustic impairments: packet loss, jitter, noise, simple PLC.

Everything takes an explicit ``numpy.random.Generator`` so a seed gives the
same output every time.
"""

from __future__ import annotations

import numpy as np

from .audio import FRAME_MS, frame_samples, resample, rms

PLC_FADE_PACKETS = 3  # repeated audio fades to silence after this many lost packets in a row


def loss_mask(n_packets: int, loss: float, rng: np.random.Generator, burst: float = 0.0) -> np.ndarray:
    """Return a boolean array, True where a packet is lost.

    ``loss`` is the target average loss rate. With ``burst`` = 0 each packet is
    lost independently. With ``burst`` in (0, 1) a two-state Gilbert-Elliott
    model is used, where ``burst`` is the probability that a loss is followed
    by another loss. The stationary loss rate is kept close to ``loss``.
    """
    if not 0.0 <= loss < 1.0:
        raise ValueError("loss must be in [0, 1)")
    if loss == 0.0 or n_packets == 0:
        return np.zeros(n_packets, dtype=bool)
    if burst <= 0.0:
        return rng.random(n_packets) < loss
    p_stay_bad = burst
    # stationary P(bad) = p_gb / (p_gb + 1 - p_stay_bad) = loss  ->  solve p_gb
    p_gb = loss * (1.0 - p_stay_bad) / (1.0 - loss)
    mask = np.zeros(n_packets, dtype=bool)
    bad = rng.random() < loss
    for i in range(n_packets):
        mask[i] = bad
        bad = rng.random() < (p_stay_bad if bad else p_gb)
    return mask


def apply_plc(frames: np.ndarray, lost: np.ndarray, mode: str = "repeat") -> np.ndarray:
    """Conceal lost packets.

    ``frames`` has shape (n_packets, samples). ``mode`` is ``none`` (silence
    insertion) or ``repeat`` (repeat the last good packet with a falling gain,
    fading to silence after a few packets). It is deliberately simple. Real
    codecs use pitch-based concealment, which sounds better.
    """
    out = frames.copy()
    if mode == "none":
        out[lost] = 0.0
        return out
    if mode != "repeat":
        raise ValueError(f"unknown PLC mode: {mode!r}")
    last_good = None
    run = 0
    for i in range(len(out)):
        if not lost[i]:
            last_good = out[i].copy()
            run = 0
            continue
        run += 1
        if last_good is None or run > PLC_FADE_PACKETS:
            out[i] = 0.0
        else:
            gain = 1.0 - run / (PLC_FADE_PACKETS + 1)
            out[i] = last_good * gain
    return out


def _to_packets(audio: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    n = frame_samples(sr, FRAME_MS)
    count = int(np.ceil(len(audio) / n))
    padded = np.pad(audio, (0, count * n - len(audio)))
    return padded.reshape(count, n), len(audio)


def packet_loss(
    audio: np.ndarray, sr: int, loss: float, rng: np.random.Generator, plc: str = "repeat", burst: float = 0.0
) -> tuple[np.ndarray, float]:
    """Drop 20 ms packets at random and conceal them. Returns audio and the realised loss rate."""
    frames, n = _to_packets(audio, sr)
    lost = loss_mask(len(frames), loss, rng, burst)
    out = apply_plc(frames, lost, plc).reshape(-1)[:n]
    realised = float(lost.mean()) if len(lost) else 0.0
    return out.astype(np.float32), realised


def jitter_loss_mask(
    n_packets: int, jitter_ms: float, buffer_ms: float, rng: np.random.Generator
) -> np.ndarray:
    """Which packets miss a fixed jitter buffer.

    Each packet gets a one-way delay drawn from |N(0, jitter_ms)|. The jitter
    buffer plays packet i at ``i * 20 ms + buffer_ms``. A packet whose delay is
    larger than ``buffer_ms`` arrives too late and is treated as lost.
    """
    if jitter_ms <= 0:
        return np.zeros(n_packets, dtype=bool)
    delay = np.abs(rng.normal(0.0, jitter_ms, n_packets))
    return delay > buffer_ms


def jitter(
    audio: np.ndarray, sr: int, jitter_ms: float, buffer_ms: float, rng: np.random.Generator, plc: str = "repeat"
) -> tuple[np.ndarray, float]:
    """Simulate delay jitter with a fixed playout buffer; late packets become loss."""
    frames, n = _to_packets(audio, sr)
    lost = jitter_loss_mask(len(frames), jitter_ms, buffer_ms, rng)
    out = apply_plc(frames, lost, plc).reshape(-1)[:n]
    return out.astype(np.float32), float(lost.mean()) if len(lost) else 0.0


def noise_signal(kind: str, n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Generate ``n`` samples of white, pink or brown noise (unit RMS)."""
    white = rng.standard_normal(n + 1024)
    if kind == "white":
        x = white[:n]
    elif kind in ("pink", "brown"):
        spec = np.fft.rfft(white)
        f = np.fft.rfftfreq(len(white), 1.0 / sr)
        f[0] = f[1]
        exponent = 0.5 if kind == "pink" else 1.0
        spec = spec / f**exponent
        x = np.fft.irfft(spec, len(white))[:n]
    else:
        raise ValueError(f"unknown noise type: {kind!r}")
    return (x / (rms(x) + 1e-12)).astype(np.float32)


def babble_signal(others: list[np.ndarray], n: int, rng: np.random.Generator) -> np.ndarray:
    """Mix other speech clips into babble noise of ``n`` samples (unit RMS)."""
    if not others:
        raise ValueError("babble needs at least one other clip")
    mix = np.zeros(n, dtype=np.float64)
    for clip in others:
        if len(clip) == 0:
            continue
        reps = int(np.ceil(n / len(clip))) + 1
        looped = np.tile(clip, reps)
        start = int(rng.integers(0, max(1, len(clip))))
        mix += looped[start : start + n]
    return (mix / (rms(mix) + 1e-12)).astype(np.float32)


def active_level(audio: np.ndarray, sr: int, floor_db: float = -40.0) -> float:
    """RMS of the louder 20 ms frames, so leading and trailing silence do not skew SNR."""
    n = frame_samples(sr, FRAME_MS)
    count = len(audio) // n
    if count == 0:
        return rms(audio)
    frames = audio[: count * n].reshape(count, n)
    levels = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))
    peak = levels.max()
    if peak <= 0:
        return 0.0
    keep = levels >= peak * 10 ** (floor_db / 20.0)
    return float(np.sqrt((levels[keep] ** 2).mean()))


def add_noise(
    audio: np.ndarray, sr: int, noise: np.ndarray, snr_db: float
) -> np.ndarray:
    """Mix ``noise`` (any level) into ``audio`` at the given SNR over active speech."""
    speech = active_level(audio, sr)
    scale = speech / (10 ** (snr_db / 20.0)) / (rms(noise) + 1e-12)
    return (audio + noise[: len(audio)] * scale).astype(np.float32)


def resample_to(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    return resample(audio, sr_in, sr_out)
