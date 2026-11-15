from __future__ import annotations

import httpx
import pytest

from gehrman.ollama import OllamaUnavailableError, list_models


def _handler(response: httpx.Response):
    def handler(request: httpx.Request) -> httpx.Response:
        return response

    return handler


def test_list_models_parses_openai_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "object": "list",
        "data": [
            {"id": "llama3.1:latest", "object": "model"},
            {"id": "digitsflow/bonsai-8b:latest", "object": "model"},
        ],
    }
    transport = httpx.MockTransport(_handler(httpx.Response(200, json=payload)))

    def fake_get(url: str, timeout: float) -> httpx.Response:
        with httpx.Client(transport=transport) as client:
            return client.get(url, timeout=timeout)

    monkeypatch.setattr("gehrman.ollama.httpx.get", fake_get)
    assert list_models("http://localhost:11434/v1") == [
        "llama3.1:latest",
        "digitsflow/bonsai-8b:latest",
    ]


def test_list_models_empty_when_no_models(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = httpx.MockTransport(
        _handler(httpx.Response(200, json={"object": "list", "data": []}))
    )

    def fake_get(url: str, timeout: float) -> httpx.Response:
        with httpx.Client(transport=transport) as client:
            return client.get(url, timeout=timeout)

    monkeypatch.setattr("gehrman.ollama.httpx.get", fake_get)
    assert list_models("http://localhost:11434/v1") == []


def test_list_models_raises_when_endpoint_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(url: str, timeout: float) -> httpx.Response:
        raise httpx.ConnectError("refused")

    monkeypatch.setattr("gehrman.ollama.httpx.get", boom)
    with pytest.raises(OllamaUnavailableError, match="refused"):
        list_models("http://localhost:11434/v1")


def test_list_models_raises_on_malformed_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = httpx.MockTransport(_handler(httpx.Response(200, json=["not", "a", "dict"])))

    def fake_get(url: str, timeout: float) -> httpx.Response:
        with httpx.Client(transport=transport) as client:
            return client.get(url, timeout=timeout)

    monkeypatch.setattr("gehrman.ollama.httpx.get", fake_get)
    with pytest.raises(OllamaUnavailableError):
        list_models("http://localhost:11434/v1")
