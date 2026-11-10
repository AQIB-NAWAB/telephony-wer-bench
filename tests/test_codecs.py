import subprocess

import numpy as np
import pytest

from twb import codecs


def _ffmpeg_decode_table(fmt: str) -> np.ndarray:
    allc = np.arange(256, dtype=np.uint8)
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        src, dst = Path(tmp) / "all.bin", Path(tmp) / "out.raw"
        allc.tofile(src)
        subprocess.run(
            ["ffmpeg", "-loglevel", "error", "-y", "-f", fmt, "-ar", "8000", "-ac", "1", "-i", str(src),
             "-f", "s16le", str(dst)],
            check=True,
        )
        return np.fromfile(dst, dtype=np.int16)


needs_ffmpeg = pytest.mark.skipif(
    subprocess.run(["which", "ffmpeg"], capture_output=True).returncode != 0, reason="ffmpeg not installed"
)


def test_mulaw_known_values():
    pcm = np.array([0, 0], dtype=np.int16)
    assert codecs.mulaw_encode(pcm).tolist() == [0xFF, 0xFF]
    assert codecs.mulaw_decode(np.array([0xFF], dtype=np.uint8)).tolist() == [0]
    assert codecs.mulaw_decode(np.array([0x00], dtype=np.uint8)).tolist() == [-32124]
    assert codecs.mulaw_decode(np.array([0x80], dtype=np.uint8)).tolist() == [32124]


def test_alaw_known_values():
    assert codecs.alaw_decode(np.array([0xD5], dtype=np.uint8)).tolist() == [8]
    assert codecs.alaw_decode(np.array([0x55], dtype=np.uint8)).tolist() == [-8]


@pytest.mark.parametrize("law", ["mulaw", "alaw"])
def test_roundtrip_error_is_bounded(law):
    enc = codecs.mulaw_encode if law == "mulaw" else codecs.alaw_encode
    dec = codecs.mulaw_decode if law == "mulaw" else codecs.alaw_decode
    pcm = np.arange(-32768, 32768, 3, dtype=np.int32).astype(np.int16)
    out = dec(enc(pcm)).astype(np.int32)
    err = np.abs(out - pcm.astype(np.int32))
    # G.711 step size is about 3% of the sample magnitude, plus a small constant at low levels
    bound = np.abs(pcm.astype(np.int32)) * 0.04 + 40
    assert (err <= bound).all()


@pytest.mark.parametrize("law", ["mulaw", "alaw"])
def test_roundtrip_is_idempotent(law):
    enc = codecs.mulaw_encode if law == "mulaw" else codecs.alaw_encode
    dec = codecs.mulaw_decode if law == "mulaw" else codecs.alaw_decode
    rng = np.random.default_rng(1)
    pcm = rng.integers(-32768, 32767, 5000).astype(np.int16)
    once = dec(enc(pcm))
    twice = dec(enc(once))
    assert (once == twice).all()


@needs_ffmpeg
@pytest.mark.parametrize("law", ["mulaw", "alaw"])
def test_decode_table_matches_ffmpeg(law):
    dec = codecs.mulaw_decode if law == "mulaw" else codecs.alaw_decode
    mine = dec(np.arange(256, dtype=np.uint8))
    assert (mine == _ffmpeg_decode_table(law)).all()


def test_g711_roundtrip_keeps_shape(speechish):
    audio, sr = speechish
    out = codecs.g711_roundtrip(audio, "mulaw")
    assert out.shape == audio.shape and out.dtype == np.float32
    assert np.abs(out - audio).max() < 0.02


def test_unknown_law():
    with pytest.raises(ValueError):
        codecs.g711_roundtrip(np.zeros(10, dtype=np.float32), "gsm")


@pytest.mark.skipif(not codecs.opus_available(), reason="ffmpeg with libopus not available")
def test_opus_roundtrip_length(speechish):
    audio, _ = speechish
    from twb.audio import resample

    x = resample(audio, 16000, 8000)
    out = codecs.opus_narrowband_roundtrip(x, 8000)
    assert len(out) == len(x)
    assert np.corrcoef(out[2000:6000], x[2000:6000])[0, 1] > 0.5
