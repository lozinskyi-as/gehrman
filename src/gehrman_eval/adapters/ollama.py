"""Ollama backend for the harness.

Thin wrapper around Gehrman's own OpenAI-compatible adapter: constructs
the client at the requested base URL, forces the model name through,
and attempts a best-effort lookup of the model tag/digest so the
report can cite exactly what ran.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx
from openai import OpenAI

from gehrman.agent.adapter import ModelAdapter, OpenAICompatAdapter

REQUEST_TIMEOUT_SECONDS = 180.0


class OllamaUnavailableError(RuntimeError):
    pass


@dataclass
class OllamaModelInfo:
    name: str
    digest: str | None = None


def strip_prefix(model_spec: str) -> str:
    """Accept either `ollama:model` or `model`; return the model name."""
    if model_spec.startswith("ollama:"):
        return model_spec[len("ollama:"):]
    return model_spec


def resolve_model_info(model_spec: str, base_url: str) -> OllamaModelInfo:
    """Best-effort lookup of the model digest; returns name only on failure."""
    name = strip_prefix(model_spec)
    # Ollama's native /api/tags lives one level up from the OpenAI base.
    native = base_url.rstrip("/").removesuffix("/v1").rstrip("/") + "/api/tags"
    try:
        response = httpx.get(native, timeout=5.0)
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError):
        return OllamaModelInfo(name=name)
    models = payload.get("models", []) if isinstance(payload, dict) else []
    for entry in models:
        if not isinstance(entry, dict):
            continue
        if entry.get("name") == name or entry.get("model") == name:
            return OllamaModelInfo(name=name, digest=entry.get("digest"))
    return OllamaModelInfo(name=name)


def build_adapter(model_spec: str, base_url: str) -> ModelAdapter:
    client = OpenAI(
        base_url=base_url,
        api_key="ollama",
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    return OpenAICompatAdapter(client, model=strip_prefix(model_spec))
