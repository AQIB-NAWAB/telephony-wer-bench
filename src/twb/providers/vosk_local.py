"""Vosk (Kaldi) running locally. It is natively streaming, with real partials."""

from __future__ import annotations

import json
import os
import time
import zipfile
from pathlib import Path

import numpy as np

from ..streamresample import to_model_rate
from .base import MissingDependency, Provider, RawEvent, Session, now

DEFAULT_MODEL = "vosk-model-small-en-us-0.15"
MODEL_URL = f"https://alphacephei.com/vosk/models/{DEFAULT_MODEL}.zip"


def _resolve_model(path: str | None) -> Path:
    path = path or os.environ.get("TWB_VOSK_MODEL")
    if path:
        return Path(path)
    cache = Path.home() / ".cache" / "twb" / DEFAULT_MODEL
    if not cache.exists():
        from ..datasets import download

        zpath = download(MODEL_URL, cache.parent / f"{DEFAULT_MODEL}.zip")
        with zipfile.ZipFile(zpath) as z:
            z.extractall(cache.parent)
    return cache


class _VoskSession(Session):
    def __init__(self, model, sample_rate: int) -> None:
        from vosk import KaldiRecognizer

        self.rec = KaldiRecognizer(model, 16000)
        self.convert = to_model_rate(sample_rate, 16000)
        self.events: list[RawEvent] = []
        self.last_partial = ""
        self.compute_seconds = 0.0

    def send(self, pcm: np.ndarray) -> None:
        data = self.convert(pcm).tobytes()
        t0 = time.perf_counter()
        is_final = self.rec.AcceptWaveform(data)
        if is_final:
            text = json.loads(self.rec.Result()).get("text", "")
            self.events.append(RawEvent("final", text, now()))
            self.last_partial = ""
        else:
            text = json.loads(self.rec.PartialResult()).get("partial", "")
            if text != self.last_partial:
                self.last_partial = text
                self.events.append(RawEvent("partial", text, now()))
        self.compute_seconds += time.perf_counter() - t0

    def poll(self) -> list[RawEvent]:
        out, self.events = self.events, []
        return out

    def finish(self) -> list[RawEvent]:
        t0 = time.perf_counter()
        text = json.loads(self.rec.FinalResult()).get("text", "")
        self.compute_seconds += time.perf_counter() - t0
        if text:
            self.events.append(RawEvent("final", text, now()))
        return self.poll()


class VoskProvider(Provider):
    name = "vosk-small-en"
    description = "Vosk small English model (Kaldi), local, natively streaming. 8 kHz input is upsampled to 16 kHz."

    def __init__(self, model_path: str | None = None) -> None:
        try:
            from vosk import Model, SetLogLevel
        except ImportError as exc:  # pragma: no cover
            raise MissingDependency('install the vosk extra: pip install "telephony-wer-bench[vosk]"') from exc
        SetLogLevel(-1)
        self.model = Model(str(_resolve_model(model_path)))

    def open(self, sample_rate: int, reference: str | None = None) -> Session:
        return _VoskSession(self.model, sample_rate)
