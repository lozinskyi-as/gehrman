"""Scripted model adapter for unit tests.

Returns a fixed list of `AdapterResponse` objects in order. Any call
beyond the script raises, so tests fail loudly if the agent loop runs
a step the test didn't plan for.
"""

from __future__ import annotations

import json
from typing import Any

from gehrman.agent.adapter import AdapterResponse, AdapterToolCall, AdapterUsage


class FakeAdapter:
    def __init__(self, scripted: list[AdapterResponse]) -> None:
        self._script = list(scripted)
        self.calls: list[dict] = []

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        temperature: float,
        max_tokens: int,
    ) -> AdapterResponse:
        self.calls.append(
            {
                "messages": list(messages),
                "tool_count": len(tools),
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if not self._script:
            raise AssertionError("FakeAdapter script exhausted")
        return self._script.pop(0)


def answer(text: str, *, tokens_in: int = 10, tokens_out: int = 5) -> AdapterResponse:
    return AdapterResponse(
        text=text,
        tool_calls=[],
        usage=AdapterUsage(tokens_in=tokens_in, tokens_out=tokens_out),
    )


def tool_call(
    name: str, arguments: dict, *, call_id: str = "call-1"
) -> AdapterResponse:
    return AdapterResponse(
        text="",
        tool_calls=[
            AdapterToolCall(id=call_id, name=name, arguments_json=json.dumps(arguments))
        ],
        usage=AdapterUsage(tokens_in=10, tokens_out=0),
    )
