"""Stream prepared audio to a provider at real-time pace and record events."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import platform
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import __version__
from .audio import FRAME_MS, iter_frames, read_audio
from .prepare import read_manifest
from .providers.base import Provider, RawEvent, Session


@dataclass
class StreamResult:
    events: list[dict]
    stream_end_t: float  # seconds after start when the last frame (incl. tail) was sent
    done_t: float  # seconds after start when finish() returned
    audio_s: float
    tail_s: float
    compute_s: float | None


def pace_frames(
    frames: list[np.ndarray],
    frame_ms: int = FRAME_MS,
    pace: bool = True,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    t0: float | None = None,
) -> Iterator[tuple[int, np.ndarray, float]]:
    """Yield (index, frame, send_time) with frame i released at start + i * frame_ms.

    Deadlines are absolute, so a slow consumer does not make the stream drift
    further behind. If it falls behind, frames are released immediately until
    it catches up. ``pace=False`` releases frames without waiting.
    """
    t0 = clock() if t0 is None else t0
    step = frame_ms / 1000.0
    for i, frame in enumerate(frames):
        if pace:
            wait = t0 + i * step - clock()
            if wait > 0:
                sleep(wait)
        yield i, frame, clock() - t0


def stream_utterance(
    session: Session,
    audio: np.ndarray,
    sr: int,
    tail_s: float = 0.6,
    pace: bool = True,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> StreamResult:
    """Send ``audio`` plus ``tail_s`` of digital silence in 20 ms frames, then finish."""
    frames = list(iter_frames(audio, sr))
    n_tail = int(round(tail_s * 1000 / FRAME_MS))
    frames += [np.zeros_like(frames[0]) for _ in range(n_tail)]
    raw: list[tuple[RawEvent, float]] = []
    t_start = clock()
    for i, frame, _ in pace_frames(frames, FRAME_MS, pace, clock, sleep, t0=t_start):
        session.send(frame)
        for ev in session.poll():
            raw.append((ev, i * FRAME_MS / 1000.0))
    stream_end = clock() - t_start
    for ev in session.finish():
        raw.append((ev, len(frames) * FRAME_MS / 1000.0))
    done = clock() - t_start
    events = [
        {"kind": ev.kind, "text": ev.text, "t": round(ev.mono - t_start, 4),
         "audio_t": round(at, 4)}
        for ev, at in raw
    ]
    events.sort(key=lambda e: e["t"])
    return StreamResult(
        events=events,
        stream_end_t=stream_end,
        done_t=done,
        audio_s=len(audio) / sr,
        tail_s=n_tail * FRAME_MS / 1000.0,
        compute_s=getattr(session, "compute_seconds", None),
    )


# ---- parallel running over a manifest ---------------------------------------

_PROVIDER: Provider | None = None
_TAIL = 0.6
_PACE = True


def _init_worker(spec: dict, tail_s: float, pace: bool) -> None:
    global _PROVIDER, _TAIL, _PACE
    from .providers import make_provider

    _PROVIDER = make_provider(spec["name"], **spec.get("options", {}))
    _TAIL, _PACE = tail_s, pace
    # warm up: first calls into a model are slow and would pollute the first utterances
    s = _PROVIDER.open(16000, reference="")
    stream_utterance(s, np.zeros(16000, dtype=np.float32), 16000, tail_s=0.2, pace=False)
    s.close()


def _run_one(entry: dict, root: str) -> dict:
    assert _PROVIDER is not None
    audio, sr = read_audio(Path(root) / entry["audio"])
    out = {k: entry[k] for k in ("id", "utterance", "condition", "text", "sample_rate", "duration_s",
                                 "speech_start_s", "speech_end_s")}
    try:
        session = _PROVIDER.open(sr, reference=entry["text"] if _PROVIDER.needs_reference else None)
        res = stream_utterance(session, audio, sr, tail_s=_TAIL, pace=_PACE)
        session.close()
        out.update(
            events=res.events,
            stream_end_t=round(res.stream_end_t, 4),
            done_t=round(res.done_t, 4),
            audio_s=round(res.audio_s, 4),
            tail_s=res.tail_s,
            compute_s=None if res.compute_s is None else round(res.compute_s, 4),
            error=None,
        )
    except Exception as exc:  # keep going, record the failure
        out.update(events=[], error=f"{type(exc).__name__}: {exc}")
    return out


def run_manifest(
    manifest: str | Path,
    provider_spec: dict,
    out_dir: str | Path,
    workers: int = 1,
    tail_s: float = 0.6,
    pace: bool = True,
    limit: int | None = None,
    only: list[str] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Run every manifest entry through the provider. Writes ``results.jsonl`` and ``run.json``."""
    mpath = Path(manifest)
    root = mpath if mpath.is_dir() else mpath.parent
    entries = read_manifest(mpath)
    if only:
        entries = [e for e in entries if e["condition"] in only]
    if limit:
        entries = entries[:limit]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Build the provider once here so model downloads happen before the workers start
    # (workers would otherwise race to download the same file) and to read its info.
    from .providers import make_provider

    info = make_provider(provider_spec["name"], **provider_spec.get("options", {})).info()
    started = time.time()
    results: dict[str, dict] = {}
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=ctx, initializer=_init_worker, initargs=(provider_spec, tail_s, pace)
    ) as pool:
        futs = {pool.submit(_run_one, e, str(root)): e["id"] for e in entries}
        for k, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results[r["id"]] = r
            if progress:
                progress(k, len(entries))
    ordered = [results[e["id"]] for e in entries]
    (out / "results.jsonl").write_text("\n".join(json.dumps(r) for r in ordered) + "\n")
    meta = {
        "twb_version": __version__,
        "provider": info,
        "provider_spec": provider_spec,
        "manifest": str(mpath),
        "utterances": len({e["utterance"] for e in entries}),
        "entries": len(entries),
        "workers": workers,
        "tail_s": tail_s,
        "paced_real_time": pace,
        "started_unix": started,
        "wall_seconds": round(time.time() - started, 1),
        "machine": {"cpus": os.cpu_count(), "platform": platform.platform(), "python": platform.python_version()},
        "errors": sum(1 for r in ordered if r.get("error")),
    }
    (out / "run.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out
