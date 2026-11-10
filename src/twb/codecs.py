"""G.711 mu-law and A-law, plus optional Opus narrowband through ffmpeg.

The G.711 coders are written in numpy from the ITU-T G.711 algorithm and are
cross-checked against ffmpeg's ``pcm_mulaw`` and ``pcm_alaw`` in the tests.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .audio import to_float, to_int16, write_wav

MULAW_BIAS = 0x84
MULAW_CLIP = 8159
_MULAW_SEG_END = np.array([0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF, 0x1FFF], dtype=np.int32)
_ALAW_SEG_END = np.array([0x1F, 0x3F, 0x7F, 0xFF, 0x1FF, 0x3FF, 0x7FF, 0xFFF], dtype=np.int32)


def _segment(val: np.ndarray, seg_end: np.ndarray) -> np.ndarray:
    """Index of the first segment whose upper bound is >= val (8 when none)."""
    return (val[..., None] > seg_end).sum(axis=-1).astype(np.int32)


def mulaw_encode(pcm: np.ndarray) -> np.ndarray:
    """Encode int16 PCM to 8-bit G.711 mu-law bytes (uint8)."""
    x = pcm.astype(np.int32) >> 2
    neg = x < 0
    mag = np.minimum(np.where(neg, -x, x), MULAW_CLIP) + (MULAW_BIAS >> 2)
    seg = _segment(mag, _MULAW_SEG_END)
    seg7 = np.minimum(seg, 7)
    uval = np.where(seg >= 8, 0x7F, (seg7 << 4) | ((mag >> (seg7 + 1)) & 0x0F))
    mask = np.where(neg, 0x7F, 0xFF)
    return (uval ^ mask).astype(np.uint8)


def mulaw_decode(codes: np.ndarray) -> np.ndarray:
    """Decode G.711 mu-law bytes to int16 PCM."""
    u = (~codes.astype(np.int32)) & 0xFF
    t = (((u & 0x0F) << 3) + MULAW_BIAS) << ((u & 0x70) >> 4)
    out = np.where(u & 0x80, MULAW_BIAS - t, t - MULAW_BIAS)
    return out.astype(np.int16)


def alaw_encode(pcm: np.ndarray) -> np.ndarray:
    """Encode int16 PCM to 8-bit G.711 A-law bytes (uint8)."""
    x = pcm.astype(np.int32) >> 3
    pos = x >= 0
    mag = np.where(pos, x, -x - 1)
    mask = np.where(pos, 0xD5, 0x55)
    seg = _segment(mag, _ALAW_SEG_END)
    aval = seg << 4
    aval = aval | np.where(seg < 2, (mag >> 1) & 0x0F, (mag >> np.minimum(seg, 7)) & 0x0F)
    aval = np.where(seg >= 8, 0x7F, aval)
    return (aval ^ mask).astype(np.uint8)


def alaw_decode(codes: np.ndarray) -> np.ndarray:
    """Decode G.711 A-law bytes to int16 PCM."""
    a = codes.astype(np.int32) ^ 0x55
    t = (a & 0x0F) << 4
    seg = (a & 0x70) >> 4
    t = np.where(seg == 0, t + 8, (t + 0x108) << np.maximum(seg - 1, 0))
    return np.where(a & 0x80, t, -t).astype(np.int16)


def g711_roundtrip(audio: np.ndarray, law: str) -> np.ndarray:
    """Encode then decode float audio through G.711. Input should be 8 kHz."""
    pcm = to_int16(audio)
    if law == "mulaw":
        return to_float(mulaw_decode(mulaw_encode(pcm)))
    if law == "alaw":
        return to_float(alaw_decode(alaw_encode(pcm)))
    raise ValueError(f"unknown G.711 law: {law!r}")


def opus_available() -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True, check=False
        ).stdout
    except OSError:
        return False
    return "libopus" in out


def opus_narrowband_roundtrip(audio: np.ndarray, sr: int = 8000, bitrate: int = 12000) -> np.ndarray:
    """Encode to Opus narrowband (8 kHz bandwidth, VoIP profile) and decode.

    Needs an ffmpeg build with libopus. The output is cut or padded to the
    input length so sample positions still line up.
    """
    if not opus_available():
        raise RuntimeError("Opus needs ffmpeg built with libopus on PATH")
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "in.wav"
        enc = Path(tmp) / "mid.opus"
        dst = Path(tmp) / "out.wav"
        write_wav(src, audio, sr)
        base = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
        subprocess.run(
            [*base, "-i", str(src), "-c:a", "libopus", "-application", "voip", "-b:a", str(bitrate),
             "-cutoff", "8000", "-frame_duration", "20", str(enc)],
            check=True,
        )
        subprocess.run([*base, "-i", str(enc), "-ar", str(sr), "-ac", "1", str(dst)], check=True)
        import soundfile as sf

        out, _ = sf.read(str(dst), dtype="float32")
    out = out.astype(np.float32)
    if len(out) < len(audio):
        out = np.pad(out, (0, len(audio) - len(out)))
    return out[: len(audio)]
