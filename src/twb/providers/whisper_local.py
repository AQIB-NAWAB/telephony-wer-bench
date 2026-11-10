"""faster-whisper running locally, with streaming simulated by re-decoding.

Whisper is not a streaming model. This provider keeps the audio received so
far and re-decodes the whole buffer every ``interval_s`` seconds of new audio,
sending each result as a partial, then sends a final decode when the stream
ends. Latency and stability numbers therefore describe this simple strategy,
not a production streaming Whisper (which would use LocalAgreement or similar).
"""

from __future__ import annotations

import queue
import threading
import time

import numpy as np

from ..streamresample import to_model_rate
from .base import MissingDependency, Provider, RawEvent, Session, now


class _WhisperSession(Session):
    def __init__(self, model, sample_rate: int, interval_s: float, beam_size: int) -> None:
        self.model = model
        self.convert = to_model_rate(sample_rate, 16000)
        self.interval = int(interval_s * 16000)
        self.beam = beam_size
        self.buf: list[np.ndarray] = []
        self.n_samples = 0
        self.decoded_upto = 0
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.events: queue.SimpleQueue[RawEvent] = queue.SimpleQueue()
        self.stop = False
        self.compute_seconds = 0.0
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _snapshot(self) -> tuple[np.ndarray, int]:
        with self.lock:
            audio = np.concatenate(self.buf) if self.buf else np.zeros(0, dtype=np.int16)
        return audio.astype(np.float32) / 32768.0, len(audio)

    def _decode(self, audio: np.ndarray) -> str:
        t0 = time.perf_counter()
        segments, _ = self.model.transcribe(
            audio,
            language="en",
            beam_size=self.beam,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=False,
            without_timestamps=True,
        )
        text = " ".join(s.text.strip() for s in segments).strip()
        self.compute_seconds += time.perf_counter() - t0
        return text

    def _loop(self) -> None:
        """All decoding runs on this one thread. Partials while audio arrives, then the final."""
        while True:
            with self.cv:
                while not self.stop and self.n_samples - self.decoded_upto < self.interval:
                    self.cv.wait(timeout=0.05)
                stopping = self.stop
            audio, n = self._snapshot()
            if stopping:
                self.events.put(RawEvent("final", self._decode(audio), now()))
                return
            text = self._decode(audio)
            self.decoded_upto = n
            if not self.stop:
                self.events.put(RawEvent("partial", text, now()))

    def send(self, pcm: np.ndarray) -> None:
        frame = self.convert(pcm)
        with self.cv:
            self.buf.append(frame)
            self.n_samples += len(frame)
            self.cv.notify()

    def poll(self) -> list[RawEvent]:
        out = []
        while True:
            try:
                out.append(self.events.get_nowait())
            except queue.Empty:
                return out

    def finish(self) -> list[RawEvent]:
        with self.cv:
            self.stop = True
            self.cv.notify_all()
        self.thread.join()
        return self.poll()

    def close(self) -> None:
        with self.cv:
            self.stop = True
            self.cv.notify_all()


class WhisperProvider(Provider):
    """compute_type defaults to float32. On the machine used for the published results, int8 gave corrupted
    transcripts (random wrong tokens) when several worker processes ran at once, while float32 did not.
    See docs/architecture.md."""

    description = "faster-whisper, local, streaming simulated by re-decoding the growing buffer."

    def __init__(
        self,
        model_size: str = "base.en",
        interval_s: float = 1.0,
        beam_size: int = 1,
        cpu_threads: int = 2,
        compute_type: str = "float32",
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover
            raise MissingDependency(
                'install the whisper extra: pip install "telephony-wer-bench[whisper]"'
            ) from exc
        self.name = f"faster-whisper-{model_size}"
        self.description = (
            f"faster-whisper {model_size} ({compute_type}), local, streaming simulated by re-decoding the "
            "growing buffer."
        )
        self.interval_s = interval_s
        self.beam_size = beam_size
        self.model = WhisperModel(model_size, device="cpu", compute_type=compute_type, cpu_threads=cpu_threads)

    def open(self, sample_rate: int, reference: str | None = None) -> Session:
        return _WhisperSession(self.model, sample_rate, self.interval_s, self.beam_size)
