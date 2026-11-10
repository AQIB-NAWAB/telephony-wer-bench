"""Test conditions: a named chain of impairments applied to clean source audio."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml

from . import codecs, impair
from .audio import resample


@dataclass
class Condition:
    name: str
    sample_rate: int = 16000
    codec: str = "none"  # none | mulaw | alaw | opus
    opus_bitrate: int = 12000
    loss: float = 0.0
    loss_burst: float = 0.0
    plc: str = "repeat"  # repeat | none
    jitter_ms: float = 0.0
    jitter_buffer_ms: float = 60.0
    noise_type: str = "none"  # none | white | pink | brown | babble
    snr_db: float | None = None
    description: str = ""
    extra: dict = field(default_factory=dict)

    def validate(self) -> None:
        if self.sample_rate not in (8000, 16000):
            raise ValueError(f"{self.name}: sample_rate must be 8000 or 16000")
        if self.codec not in ("none", "mulaw", "alaw", "opus"):
            raise ValueError(f"{self.name}: unknown codec {self.codec!r}")
        if self.codec != "none" and self.sample_rate != 8000:
            raise ValueError(f"{self.name}: codec {self.codec} needs sample_rate 8000")
        if self.noise_type != "none" and self.snr_db is None:
            raise ValueError(f"{self.name}: noise needs snr_db")
        if self.noise_type not in ("none", "white", "pink", "brown", "babble"):
            raise ValueError(f"{self.name}: unknown noise type {self.noise_type!r}")
        if not 0.0 <= self.loss < 1.0:
            raise ValueError(f"{self.name}: loss must be in [0, 1)")

    def params(self) -> dict:
        d = asdict(self)
        d.pop("extra")
        return d


def condition_from_dict(d: dict) -> Condition:
    d = dict(d)
    noise = d.pop("noise", None)
    if noise:
        d["noise_type"] = noise.get("type", "pink")
        d["snr_db"] = noise.get("snr_db")
    known = set(Condition.__dataclass_fields__)
    extra = {k: d.pop(k) for k in list(d) if k not in known}
    c = Condition(**d, extra=extra)
    c.validate()
    return c


def load_conditions(path: str | Path) -> list[Condition]:
    data = yaml.safe_load(Path(path).read_text())
    items = data["conditions"] if isinstance(data, dict) else data
    conds = [condition_from_dict(x) for x in items]
    names = [c.name for c in conds]
    if len(set(names)) != len(names):
        raise ValueError("condition names must be unique")
    return conds


def derive_seed(base_seed: int, *parts: str) -> int:
    """Stable 63-bit seed from a base seed and string parts (not Python's salted hash)."""
    h = hashlib.sha256(("|".join([str(base_seed), *parts])).encode()).digest()
    return int.from_bytes(h[:8], "big") >> 1


@dataclass
class Applied:
    audio: np.ndarray
    sample_rate: int
    realised_loss: float = 0.0


def apply_condition(
    cond: Condition,
    audio: np.ndarray,
    sr: int,
    seed: int,
    babble_sources: list[np.ndarray] | None = None,
) -> Applied:
    """Run clean ``audio`` (any rate) through ``cond``. The output is at ``cond.sample_rate``."""
    rng = np.random.default_rng(seed)
    x = audio.astype(np.float32)
    # 1. acoustic noise, added where the caller is, before any band limiting
    if cond.noise_type != "none":
        if cond.noise_type == "babble":
            noise = impair.babble_signal(babble_sources or [], len(x), rng)
        else:
            noise = impair.noise_signal(cond.noise_type, len(x), sr, rng)
        x = impair.add_noise(x, sr, noise, float(cond.snr_db))
    # 2. sample rate
    x = resample(x, sr, cond.sample_rate)
    # 3. codec
    if cond.codec in ("mulaw", "alaw"):
        x = codecs.g711_roundtrip(x, cond.codec)
    elif cond.codec == "opus":
        x = codecs.opus_narrowband_roundtrip(x, cond.sample_rate, cond.opus_bitrate)
    # 4. network: jitter-buffer misses, then random loss, both concealed
    realised = 0.0
    if cond.jitter_ms > 0:
        x, jl = impair.jitter(x, cond.sample_rate, cond.jitter_ms, cond.jitter_buffer_ms, rng, cond.plc)
        realised = jl
    if cond.loss > 0:
        x, rl = impair.packet_loss(x, cond.sample_rate, cond.loss, rng, cond.plc, cond.loss_burst)
        realised = 1.0 - (1.0 - realised) * (1.0 - rl)
    return Applied(np.clip(x, -1.0, 1.0).astype(np.float32), cond.sample_rate, realised)
