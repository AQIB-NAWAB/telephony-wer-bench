"""Shared plumbing for WebSocket adapters: a receive thread feeding an event queue."""

from __future__ import annotations

import contextlib
import json
import queue
import threading

from .base import MissingDependency, RawEvent, Session, now


def _connect(url: str, headers: dict[str, str]):
    try:
        from websockets.sync.client import connect
    except ImportError as exc:  # pragma: no cover
        raise MissingDependency('install the cloud extra: pip install "telephony-wer-bench[cloud]"') from exc
    return connect(url, additional_headers=headers, max_size=None, open_timeout=15)


class WebSocketSession(Session):
    """Subclasses set ``url``/``headers`` and implement ``encode``, ``parse`` and ``end_messages``."""

    url = ""
    close_wait_s = 15.0

    def __init__(self, headers: dict[str, str]) -> None:
        self.ws = _connect(self.url, headers)
        self.events: queue.SimpleQueue[RawEvent] = queue.SimpleQueue()
        self.done = threading.Event()
        self.error: str | None = None
        self.thread = threading.Thread(target=self._recv, daemon=True)
        self.on_open()
        self.thread.start()

    # hooks ------------------------------------------------------------
    def on_open(self) -> None:  # noqa: B027
        """Send any setup messages."""

    def encode(self, pcm) -> bytes | str:  # pragma: no cover - interface
        raise NotImplementedError

    def parse(self, message: dict) -> list[tuple[str, str]]:  # pragma: no cover - interface
        """Map one server message to (kind, text) events. Return [] to ignore it."""
        raise NotImplementedError

    def end_messages(self) -> list[bytes | str]:  # pragma: no cover - interface
        raise NotImplementedError

    def is_last(self, message: dict) -> bool:
        return False

    # plumbing ---------------------------------------------------------
    def _recv(self) -> None:
        try:
            for raw in self.ws:
                if isinstance(raw, bytes):
                    continue
                msg = json.loads(raw)
                stamp = now()
                for kind, text in self.parse(msg):
                    self.events.put(RawEvent(kind, text, stamp))
                if self.is_last(msg):
                    break
        except Exception as exc:  # connection closed or bad message
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.done.set()

    def send(self, pcm) -> None:
        self.ws.send(self.encode(pcm))

    def poll(self) -> list[RawEvent]:
        out = []
        while True:
            try:
                out.append(self.events.get_nowait())
            except queue.Empty:
                return out

    def finish(self) -> list[RawEvent]:
        for m in self.end_messages():
            self.ws.send(m)
        self.done.wait(self.close_wait_s)
        out = self.poll()
        self.close()
        return out

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.ws.close()
