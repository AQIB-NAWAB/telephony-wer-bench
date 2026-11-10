"""Provider interface.

A provider opens a streaming ``Session`` for one utterance. The runner calls
``send`` with 20 ms int16 frames at real-time pace, ``poll`` after each frame
and ``finish`` once after the last frame.

Event contract: ``partial`` text is the current best guess for the segment
that has not been finalized yet. ``final`` text is a segment that will not
change. A provider that never finalizes mid-stream sends one final from
``finish``. Timestamps are ``time.monotonic()`` taken when the event arrived.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class RawEvent:
    kind: str  # "partial" or "final"
    text: str
    mono: float  # time.monotonic() at arrival


def now() -> float:
    return time.monotonic()


class Session(ABC):
    #: seconds of provider compute spent, when the provider can measure it (local models)
    compute_seconds: float | None = None

    @abstractmethod
    def send(self, pcm: np.ndarray) -> None:
        """Accept one 20 ms int16 mono frame at the session sample rate."""

    @abstractmethod
    def poll(self) -> list[RawEvent]:
        """Return events that arrived since the last call. Must not block."""

    @abstractmethod
    def finish(self) -> list[RawEvent]:
        """Signal end of audio, wait for final results and return the remaining events."""

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release resources."""


class Provider(ABC):
    name: str = "provider"
    #: set True only by the debug provider that reads the reference text
    needs_reference: bool = False
    #: True for adapters that were written from public docs and never run
    tested: bool = True
    description: str = ""

    @abstractmethod
    def open(self, sample_rate: int, reference: str | None = None) -> Session:
        """Start a streaming session for audio at ``sample_rate`` (8000 or 16000)."""

    def info(self) -> dict:
        return {"name": self.name, "tested": self.tested, "description": self.description}


class MissingDependency(RuntimeError):
    pass


def require_env(var: str, provider: str) -> str:
    val = os.environ.get(var)
    if not val:
        raise MissingDependency(f"{provider}: set the {var} environment variable")
    return val
