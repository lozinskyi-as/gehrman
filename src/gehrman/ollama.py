"""Query a local Ollama OpenAI-compatible endpoint for installed models.

Used by `gehrman init` to let the researcher pick a model from what they
have already pulled. Loopback-only; the call must be fast and must not
hang the install flow if Ollama is down.
"""

from __future__ import annotations

import httpx

DISCOVERY_TIMEOUT_SECONDS = 5.0


class OllamaUnavailableError(RuntimeError):
    """Raised when the Ollama endpoint cannot be reached or parsed."""


def list_models(base_url: str, *, timeout: float = DISCOVERY_TIMEOUT_SECONDS) -> list[str]:
    """Return locally installed model identifiers.

    An empty list means Ollama is reachable but has no models pulled.
    """
    url = base_url.rstrip("/") + "/models"
    try:
        response = httpx.get(url, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise OllamaUnavailableError(str(exc)) from exc
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise OllamaUnavailableError(f"Unexpected response from {url}")
    models: list[str] = []
    for entry in data:
        if isinstance(entry, dict):
            model_id = entry.get("id")
            if isinstance(model_id, str) and model_id:
                models.append(model_id)
    return models
