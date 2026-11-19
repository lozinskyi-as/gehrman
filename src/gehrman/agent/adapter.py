"""Model adapter so the loop can talk to different LLM backends.

Gehrman's chat uses the default `OpenAICompatAdapter` pointed at the
local Ollama endpoint. The thesis evaluation harness swaps in its own
adapter (Anthropic, or a scripted fake) without Gehrman itself importing
`anthropic` or adding any new network code.

The adapter surface is deliberately narrow: one `complete` call that
returns text, requested tool calls, and token usage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from openai import OpenAI, OpenAIError


class AdapterError(RuntimeError):
    """Raised when the underlying model backend call fails."""


@dataclass
class AdapterToolCall:
    """One tool invocation requested by the model."""

    id: str
    name: str
    arguments_json: str


@dataclass
class AdapterUsage:
    tokens_in: int = 0
    tokens_out: int = 0


@dataclass
class AdapterResponse:
    text: str
    tool_calls: list[AdapterToolCall] = field(default_factory=list)
    usage: AdapterUsage = field(default_factory=AdapterUsage)


class ModelAdapter(Protocol):
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> AdapterResponse: ...


class OpenAICompatAdapter:
    """Default adapter: OpenAI SDK pointed at Ollama's OpenAI-compatible endpoint.

    Preserves the exact request shape the Gehrman MVP shipped with.
    """

    def __init__(self, client: OpenAI, model: str) -> None:
        self._client = client
        self._model = model

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> AdapterResponse:
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except OpenAIError as exc:
            raise AdapterError(f"Model call failed: {exc}") from exc

        message = response.choices[0].message
        tool_calls: list[AdapterToolCall] = []
        if message.tool_calls:
            for call in message.tool_calls:
                tool_calls.append(
                    AdapterToolCall(
                        id=call.id,
                        name=call.function.name,
                        arguments_json=call.function.arguments or "",
                    )
                )
        usage = getattr(response, "usage", None)
        tokens_in = int(getattr(usage, "prompt_tokens", 0) or 0)
        tokens_out = int(getattr(usage, "completion_tokens", 0) or 0)
        return AdapterResponse(
            text=message.content or "",
            tool_calls=tool_calls,
            usage=AdapterUsage(tokens_in=tokens_in, tokens_out=tokens_out),
        )
