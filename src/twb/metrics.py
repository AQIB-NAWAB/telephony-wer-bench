"""Accuracy, latency and stability metrics. Each one is defined in docs/metrics.md."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean

import numpy as np

from .normalize import normalize


@dataclass
class EditCounts:
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    ref_len: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        if self.ref_len == 0:
            return 0.0 if self.errors == 0 else float("inf")
        return self.errors / self.ref_len

    def __add__(self, other: EditCounts) -> EditCounts:
        return EditCounts(
            self.substitutions + other.substitutions,
            self.deletions + other.deletions,
            self.insertions + other.insertions,
            self.ref_len + other.ref_len,
        )


def edit_counts(ref: list[str], hyp: list[str]) -> EditCounts:
    """Levenshtein alignment counts between two token lists."""
    n, m = len(ref), len(hyp)
    # dp[i][j] = (cost, S, D, I)
    prev = [(j, 0, 0, j) for j in range(m + 1)]
    for i in range(1, n + 1):
        cur = [(i, 0, i, 0)]
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                cur.append(prev[j - 1])
                continue
            c_sub, s, d, ins = prev[j - 1]
            c_del, s2, d2, i2 = prev[j]
            c_ins, s3, d3, i3 = cur[j - 1]
            options = [
                (c_sub + 1, s + 1, d, ins),
                (c_del + 1, s2, d2 + 1, i2),
                (c_ins + 1, s3, d3, i3 + 1),
            ]
            cur.append(min(options, key=lambda o: o[0]))
        prev = cur
    _, s, d, i = prev[m]
    return EditCounts(s, d, i, n)


def wer_counts(ref: str, hyp: str, normalizer=normalize) -> EditCounts:
    return edit_counts(normalizer(ref).split(), normalizer(hyp).split())


def cer_counts(ref: str, hyp: str, normalizer=normalize) -> EditCounts:
    """Character error counts. Spaces are kept as characters, collapsed to single spaces."""
    return edit_counts(list(normalizer(ref)), list(normalizer(hyp)))


def wer(ref: str, hyp: str) -> float:
    return wer_counts(ref, hyp).rate


def cer(ref: str, hyp: str) -> float:
    return cer_counts(ref, hyp).rate


@dataclass
class Event:
    """One transcript event from a provider.

    ``t`` is wall seconds since the first audio frame was sent. ``audio_t`` is
    the audio position (seconds) that had been sent when the event was seen.
    """

    kind: str  # "partial" or "final"
    text: str
    t: float
    audio_t: float = 0.0

    def to_dict(self) -> dict:
        return {"kind": self.kind, "text": self.text, "t": round(self.t, 4), "audio_t": round(self.audio_t, 4)}

    @classmethod
    def from_dict(cls, d: dict) -> Event:
        return cls(d["kind"], d["text"], float(d["t"]), float(d.get("audio_t", 0.0)))


def final_text(events: list[Event]) -> str:
    """Join all final segments. If the provider sent none, fall back to the last partial."""
    finals = [e.text for e in events if e.kind == "final" and e.text.strip()]
    if finals:
        return " ".join(finals)
    partials = [e.text for e in events if e.kind == "partial" and e.text.strip()]
    return partials[-1] if partials else ""


def time_to_first_partial(events: list[Event], speech_start: float) -> float | None:
    """Seconds from the start of speech to the first non-empty partial (or final)."""
    for e in events:
        if e.text.strip():
            return max(0.0, e.t - speech_start)
    return None


def final_latency(events: list[Event], speech_end: float) -> float | None:
    """Seconds from the end of speech to the last final event with text."""
    finals = [e for e in events if e.kind == "final" and e.text.strip()]
    if not finals:
        return None
    return finals[-1].t - speech_end


def _common_prefix(a: list[str], b: list[str]) -> int:
    k = 0
    for x, y in zip(a, b):
        if x != y:
            break
        k += 1
    return k


@dataclass
class Stability:
    partial_count: int = 0
    words_shown: int = 0
    words_retracted: int = 0
    changed_updates: int = 0
    updates: int = 0

    @property
    def retraction_rate(self) -> float:
        return self.words_retracted / self.words_shown if self.words_shown else 0.0

    @property
    def flicker_rate(self) -> float:
        """Share of partial updates that changed words that were already shown."""
        return self.changed_updates / self.updates if self.updates else 0.0

    def __add__(self, other: Stability) -> Stability:
        return Stability(
            self.partial_count + other.partial_count,
            self.words_shown + other.words_shown,
            self.words_retracted + other.words_retracted,
            self.changed_updates + other.changed_updates,
            self.updates + other.updates,
        )


def cumulative_views(events: list[Event], normalizer=normalize) -> list[list[str]]:
    """Rebuild what a caller would see: finalized segments plus the current partial.

    Consecutive identical views are collapsed.
    """
    committed: list[str] = []
    views: list[list[str]] = []
    for e in events:
        if e.kind not in ("partial", "final"):
            continue
        view = committed + normalizer(e.text).split()
        if e.kind == "final":
            committed = view
        if not views or views[-1] != view:
            views.append(view)
    return views


def partial_stability(events: list[Event], normalizer=normalize) -> Stability:
    """Measure how much shown words are rewritten by later updates.

    Walk consecutive views a -> b (see ``cumulative_views``). The words of
    ``a`` after their longest common prefix with ``b`` were shown and then
    taken back or rewritten. ``retraction_rate`` = retracted words / words
    shown. ``flicker_rate`` = updates that retracted anything / all updates.
    """
    views = cumulative_views(events, normalizer)
    st = Stability(partial_count=sum(1 for e in events if e.kind == "partial"))
    for a, b in zip(views, views[1:]):
        st.updates += 1
        st.words_shown += len(a)
        retracted = len(a) - _common_prefix(a, b)
        st.words_retracted += retracted
        if retracted > 0:
            st.changed_updates += 1
    return st


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


@dataclass
class Summary:
    """Aggregated metrics for a group of utterances (one provider, one condition)."""

    n: int = 0
    words: int = 0
    wer_counts: EditCounts = field(default_factory=EditCounts)
    cer_counts: EditCounts = field(default_factory=EditCounts)
    ttfp: list[float] = field(default_factory=list)
    final_lat: list[float] = field(default_factory=list)
    flush: list[float] = field(default_factory=list)
    rtf: list[float] = field(default_factory=list)
    stability: Stability = field(default_factory=Stability)
    empty: int = 0

    def to_dict(self) -> dict:
        def stats(vals: list[float]) -> dict | None:
            if not vals:
                return None
            return {
                "mean_ms": round(mean(vals) * 1000, 1),
                "p50_ms": round(percentile(vals, 50) * 1000, 1),
                "p95_ms": round(percentile(vals, 95) * 1000, 1),
                "n": len(vals),
            }

        return {
            "utterances": self.n,
            "ref_words": self.wer_counts.ref_len,
            "wer": round(self.wer_counts.rate, 4),
            "wer_sub": self.wer_counts.substitutions,
            "wer_del": self.wer_counts.deletions,
            "wer_ins": self.wer_counts.insertions,
            "cer": round(self.cer_counts.rate, 4),
            "empty_transcripts": self.empty,
            "time_to_first_partial": stats(self.ttfp),
            "final_latency_after_speech_end": stats(self.final_lat),
            "flush_after_stream_end": stats(self.flush),
            "real_time_factor": round(mean(self.rtf), 3) if self.rtf else None,
            "partials_total": self.stability.partial_count,
            "retraction_rate": round(self.stability.retraction_rate, 4),
            "flicker_rate": round(self.stability.flicker_rate, 4),
        }
