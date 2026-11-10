"""Provider registry."""

from __future__ import annotations

from .base import MissingDependency, Provider, RawEvent, Session

PROVIDERS = {
    "vosk": ("twb.providers.vosk_local", "VoskProvider"),
    "whisper": ("twb.providers.whisper_local", "WhisperProvider"),
    "oracle": ("twb.providers.oracle", "OracleProvider"),
    "deepgram": ("twb.providers.deepgram", "DeepgramProvider"),
    "assemblyai": ("twb.providers.assemblyai", "AssemblyAIProvider"),
    "openai-realtime": ("twb.providers.openai_realtime", "OpenAIRealtimeProvider"),
    "google": ("twb.providers.google", "GoogleProvider"),
    "azure": ("twb.providers.azure", "AzureProvider"),
}


def make_provider(name: str, **options) -> Provider:
    import importlib

    if name not in PROVIDERS:
        raise KeyError(f"unknown provider {name!r}. Known: {', '.join(sorted(PROVIDERS))}")
    module, cls = PROVIDERS[name]
    return getattr(importlib.import_module(module), cls)(**options)


__all__ = ["PROVIDERS", "MissingDependency", "Provider", "RawEvent", "Session", "make_provider"]
