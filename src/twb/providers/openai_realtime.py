"""OpenAI Realtime transcription session. UNTESTED: written from the public docs, never run.

Set OPENAI_API_KEY. 8 kHz audio is sent as G.711 mu-law (``audio/pcmu``), the
format a phone line already uses. 16 kHz audio is resampled to 24 kHz PCM.
The session has no server VAD, so the buffer is committed when the stream ends,
which means this adapter reports no mid-stream finals.
https://developers.openai.com/api/docs/guides/realtime-transcription
"""

from __future__ import annotations

import base64
import json

import numpy as np

from ..audio import resample, to_float, to_int16
from ..codecs import mulaw_encode
from ._ws import WebSocketSession
from .base import Provider, require_env


class _Session(WebSocketSession):
    url = "wss://api.openai.com/v1/realtime?intent=transcription"

    def __init__(self, key: str, sample_rate: int, model: str) -> None:
        self.sample_rate = sample_rate
        self.model = model
        self.acc = ""
        super().__init__({"Authorization": f"Bearer {key}"})

    def on_open(self) -> None:
        fmt = {"type": "audio/pcmu"} if self.sample_rate == 8000 else {"type": "audio/pcm", "rate": 24000}
        self.ws.send(json.dumps({
            "type": "session.update",
            "session": {"type": "transcription", "audio": {"input": {
                "format": fmt, "transcription": {"model": self.model}, "turn_detection": None}}},
        }))

    def encode(self, pcm):
        if self.sample_rate == 8000:
            raw = mulaw_encode(pcm).tobytes()
        else:
            up = resample(to_float(pcm), self.sample_rate, 24000)
            raw = to_int16(np.asarray(up)).tobytes()
        return json.dumps({"type": "input_audio_buffer.append", "audio": base64.b64encode(raw).decode()})

    def parse(self, message):
        t = message.get("type")
        if t == "conversation.item.input_audio_transcription.delta":
            self.acc += message.get("delta", "")
            return [("partial", self.acc)]
        if t == "conversation.item.input_audio_transcription.completed":
            self.acc = ""
            return [("final", message.get("transcript", ""))]
        return []

    def end_messages(self):
        return [json.dumps({"type": "input_audio_buffer.commit"})]

    def is_last(self, message):
        return message.get("type") == "conversation.item.input_audio_transcription.completed"


class OpenAIRealtimeProvider(Provider):
    name = "openai-realtime"
    tested = False
    description = "OpenAI Realtime transcription. Adapter not run yet."

    def __init__(self, model: str = "gpt-live-transcribe") -> None:
        self.key = require_env("OPENAI_API_KEY", "openai-realtime")
        self.model = model
        self.name = f"openai-{model}"

    def open(self, sample_rate: int, reference: str | None = None):
        return _Session(self.key, sample_rate, self.model)
