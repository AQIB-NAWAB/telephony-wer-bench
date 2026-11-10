"""Azure AI Speech streaming. UNTESTED: written from the public docs, never run.

Needs ``pip install azure-cognitiveservices-speech`` and the env vars
AZURE_SPEECH_KEY and AZURE_SPEECH_REGION. Audio goes out as 16-bit PCM.
"""

from __future__ import annotations

import queue
import threading

from .base import MissingDependency, Provider, RawEvent, Session, now, require_env


class _Session(Session):
    def __init__(self, sdk, key: str, region: str, sample_rate: int) -> None:
        self.events: queue.SimpleQueue[RawEvent] = queue.SimpleQueue()
        self.done = threading.Event()
        cfg = sdk.SpeechConfig(subscription=key, region=region)
        fmt = sdk.audio.AudioStreamFormat(samples_per_second=sample_rate, bits_per_sample=16, channels=1)
        self.stream = sdk.audio.PushAudioInputStream(stream_format=fmt)
        audio = sdk.audio.AudioConfig(stream=self.stream)
        self.rec = sdk.SpeechRecognizer(speech_config=cfg, audio_config=audio)
        self.rec.recognizing.connect(lambda e: self.events.put(RawEvent("partial", e.result.text, now())))
        self.rec.recognized.connect(lambda e: self.events.put(RawEvent("final", e.result.text, now())))
        self.rec.session_stopped.connect(lambda e: self.done.set())
        self.rec.canceled.connect(lambda e: self.done.set())
        self.rec.start_continuous_recognition_async().get()

    def send(self, pcm) -> None:
        self.stream.write(pcm.tobytes())

    def poll(self) -> list[RawEvent]:
        out = []
        while True:
            try:
                out.append(self.events.get_nowait())
            except queue.Empty:
                return out

    def finish(self) -> list[RawEvent]:
        self.stream.close()
        self.done.wait(20)
        self.rec.stop_continuous_recognition_async().get()
        return self.poll()


class AzureProvider(Provider):
    name = "azure"
    tested = False
    description = "Azure AI Speech streaming. Adapter not run yet."

    def __init__(self) -> None:
        try:
            import azure.cognitiveservices.speech as sdk
        except ImportError as exc:
            raise MissingDependency("pip install azure-cognitiveservices-speech") from exc
        self.sdk = sdk
        self.key = require_env("AZURE_SPEECH_KEY", "azure")
        self.region = require_env("AZURE_SPEECH_REGION", "azure")

    def open(self, sample_rate: int, reference: str | None = None):
        return _Session(self.sdk, self.key, self.region, sample_rate)
