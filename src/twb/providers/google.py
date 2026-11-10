"""Google Cloud Speech-to-Text v1 streaming. UNTESTED: written from the public docs, never run.

Needs ``pip install google-cloud-speech`` and Application Default Credentials
(GOOGLE_APPLICATION_CREDENTIALS). Audio goes out as LINEAR16 at the test set's rate.
"""

from __future__ import annotations

import contextlib
import queue
import threading

from .base import MissingDependency, Provider, RawEvent, Session, now


class _Session(Session):
    def __init__(self, client, speech, sample_rate: int, model: str) -> None:
        self.speech = speech
        self.audio_q: queue.Queue[bytes | None] = queue.Queue()
        self.events: queue.SimpleQueue[RawEvent] = queue.SimpleQueue()
        config = speech.StreamingRecognitionConfig(
            config=speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=sample_rate,
                language_code="en-US",
                model=model,
            ),
            interim_results=True,
        )

        def requests():
            while (chunk := self.audio_q.get()) is not None:
                yield speech.StreamingRecognizeRequest(audio_content=chunk)

        def run():
            with contextlib.suppress(Exception):
                for resp in client.streaming_recognize(config, requests()):
                    for r in resp.results:
                        if r.alternatives:
                            kind = "final" if r.is_final else "partial"
                            self.events.put(RawEvent(kind, r.alternatives[0].transcript, now()))

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def send(self, pcm) -> None:
        self.audio_q.put(pcm.tobytes())

    def poll(self) -> list[RawEvent]:
        out = []
        while True:
            try:
                out.append(self.events.get_nowait())
            except queue.Empty:
                return out

    def finish(self) -> list[RawEvent]:
        self.audio_q.put(None)
        self.thread.join(timeout=20)
        return self.poll()


class GoogleProvider(Provider):
    name = "google"
    tested = False
    description = "Google Cloud Speech-to-Text v1 streaming. Adapter not run yet."

    def __init__(self, model: str = "telephony") -> None:
        try:
            from google.cloud import speech
        except ImportError as exc:
            raise MissingDependency("pip install google-cloud-speech") from exc
        self.speech = speech
        self.client = speech.SpeechClient()
        self.model = model
        self.name = f"google-{model}"

    def open(self, sample_rate: int, reference: str | None = None):
        return _Session(self.client, self.speech, sample_rate, self.model)
