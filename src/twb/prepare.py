"""Turn a source set into a telephony test set with a JSONL manifest."""

from __future__ import annotations

import json
from pathlib import Path

from . import __version__
from .audio import read_audio, speech_bounds, write_wav
from .conditions import Condition, apply_condition, derive_seed
from .datasets import SourceItem


def prepare(
    items: list[SourceItem],
    conditions: list[Condition],
    out_dir: str | Path,
    seed: int = 1234,
    source_name: str = "",
) -> Path:
    """Write one WAV per (utterance, condition) and ``manifest.jsonl``. Returns the manifest path.

    The manifest has one JSON object per line. ``speech_start`` and ``speech_end``
    come from the clean source audio so that latency is measured against the
    same reference points in every condition.
    """
    out = Path(out_dir)
    audio_dir = out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    clean = {it.id: read_audio(it.audio) for it in items}
    lines = []
    for it in items:
        audio, sr = clean[it.id]
        start, end = speech_bounds(audio, sr)
        others = [clean[o.id][0] for o in items if o.id != it.id]
        # babble uses a few other speakers, picked deterministically
        others.sort(key=lambda a: derive_seed(seed, it.id, str(len(a))))
        for cond in conditions:
            s = derive_seed(seed, it.id, cond.name)
            res = apply_condition(cond, audio, sr, s, babble_sources=others[:4])
            rel = f"audio/{cond.name}/{it.id}.wav"
            write_wav(out / rel, res.audio, res.sample_rate)
            lines.append(
                json.dumps(
                    {
                        "id": f"{it.id}@{cond.name}",
                        "utterance": it.id,
                        "condition": cond.name,
                        "audio": rel,
                        "text": it.text,
                        "sample_rate": res.sample_rate,
                        "duration_s": round(len(res.audio) / res.sample_rate, 4),
                        "speech_start_s": round(start, 3),
                        "speech_end_s": round(end, 3),
                        "seed": s,
                        "realised_loss": round(res.realised_loss, 4),
                        "params": cond.params(),
                    }
                )
            )
    manifest = out / "manifest.jsonl"
    manifest.write_text("\n".join(lines) + "\n")
    meta = {
        "twb_version": __version__,
        "base_seed": seed,
        "source": source_name,
        "utterances": len(items),
        "conditions": [c.params() for c in conditions],
    }
    (out / "prepare.json").write_text(json.dumps(meta, indent=2) + "\n")
    return manifest


def read_manifest(path: str | Path) -> list[dict]:
    p = Path(path)
    if p.is_dir():
        p = p / "manifest.jsonl"
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
