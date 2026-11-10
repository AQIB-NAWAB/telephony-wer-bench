"""Deepgram live streaming (WebSocket). UNTESTED: written from the public docs, never run.

Set DEEPGRAM_API_KEY. Audio goes out as linear16 PCM at the test set's sample rate.
https://developers.deepgram.com/reference/speech-to-text/listen-streaming
"""

from __future__ import annotations

import json

from ._ws import WebSocketSession
from .base import Provider, require_env


class _Session(WebSocketSession):
    def __init__(self, key: str, sample_rate: int, model: str, endpointing_ms: int) -> None:
        self.url = (
            "wss://api.deepgram.com/v1/listen?encoding=linear16"
            f"&sample_rate={sample_rate}&channels=1&model={model}"
            f"&interim_results=true&endpointing={endpointing_ms}&punctuate=false"
        )
        super().__init__({"Authorization": f"Token {key}"})

    def encode(self, pcm):
        return pcm.tobytes()

    def parse(self, message):
        if message.get("type") != "Results":
            return []
        alts = message.get("channel", {}).get("alternatives", [])
        text = alts[0].get("transcript", "") if alts else ""
        return [("final" if message.get("is_final") else "partial", text)]

    def end_messages(self):
        return [json.dumps({"type": "CloseStream"})]


class DeepgramProvider(Provider):
    name = "deepgram"
    tested = False
    description = "Deepgram live streaming API. Adapter not run yet."

    def __init__(self, model: str = "nova-3", endpointing_ms: int = 300) -> None:
        self.key = require_env("DEEPGRAM_API_KEY", "deepgram")
        self.model = model
        self.endpointing_ms = endpointing_ms
        self.name = f"deepgram-{model}"

    def open(self, sample_rate: int, reference: str | None = None):
        return _Session(self.key, sample_rate, self.model, self.endpointing_ms)
