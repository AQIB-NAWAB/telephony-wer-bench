"""AssemblyAI streaming v3 (WebSocket). UNTESTED: written from the public docs, never run.

Set ASSEMBLYAI_API_KEY. Audio goes out as pcm_s16le at the test set's sample rate.
https://www.assemblyai.com/docs/api-reference/streaming-api/streaming-api
"""

from __future__ import annotations

import json

from ._ws import WebSocketSession
from .base import Provider, require_env


class _Session(WebSocketSession):
    def __init__(self, key: str, sample_rate: int, speech_model: str) -> None:
        self.url = (
            "wss://streaming.assemblyai.com/v3/ws"
            f"?sample_rate={sample_rate}&encoding=pcm_s16le&speech_model={speech_model}"
        )
        super().__init__({"Authorization": key})

    def encode(self, pcm):
        return pcm.tobytes()

    def parse(self, message):
        if message.get("type") != "Turn":
            return []
        words = message.get("words") or []
        text = " ".join(w.get("text", "") for w in words) or message.get("transcript", "")
        return [("final" if message.get("end_of_turn") else "partial", text)]

    def end_messages(self):
        return [json.dumps({"type": "Terminate"})]

    def is_last(self, message):
        return message.get("type") == "Termination"


class AssemblyAIProvider(Provider):
    name = "assemblyai"
    tested = False
    description = "AssemblyAI streaming v3. Adapter not run yet."

    def __init__(self, speech_model: str = "universal-streaming-english") -> None:
        self.key = require_env("ASSEMBLYAI_API_KEY", "assemblyai")
        self.speech_model = speech_model
        self.name = f"assemblyai-{speech_model}"

    def open(self, sample_rate: int, reference: str | None = None):
        return _Session(self.key, sample_rate, self.speech_model)
