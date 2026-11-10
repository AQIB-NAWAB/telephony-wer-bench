"""Debug provider that cheats: it reads the reference text.

It reveals the reference one word at a time as audio arrives, so the whole
pipeline (pacing, events, metrics, reports) can be tested without a model or
network. Its WER is zero by construction. Never report its numbers as ASR results.
"""

from __future__ import annotations

import numpy as np

from .base import Provider, RawEvent, Session, now


class _OracleSession(Session):
    def __init__(self, reference: str, sample_rate: int, words_per_second: float) -> None:
        self.words = reference.split()
        self.per_frame = words_per_second * 0.02
        self.sent = 0.0
        self.shown = 0
        self.events: list[RawEvent] = []

    def send(self, pcm: np.ndarray) -> None:
        self.sent += self.per_frame
        target = min(len(self.words), int(self.sent))
        if target > self.shown:
            self.shown = target
            self.events.append(RawEvent("partial", " ".join(self.words[:target]), now()))

    def poll(self) -> list[RawEvent]:
        out, self.events = self.events, []
        return out

    def finish(self) -> list[RawEvent]:
        return [*self.poll(), RawEvent("final", " ".join(self.words), now())]


class OracleProvider(Provider):
    name = "oracle"
    needs_reference = True
    description = "Debug only. Reads the reference text. Not an ASR result."

    def __init__(self, words_per_second: float = 2.5) -> None:
        self.wps = words_per_second

    def open(self, sample_rate: int, reference: str | None = None) -> Session:
        return _OracleSession(reference or "", sample_rate, self.wps)
