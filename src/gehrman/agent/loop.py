"""Tool-use loop against a local Ollama model via the OpenAI SDK.

The loop is bounded by `agent.max_tool_steps` so a stuck model cannot
spend tokens forever. Temperature is fixed at 0 for reproducibility.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from openai import OpenAI, OpenAIError

from gehrman.agent.prompts import SYSTEM_PROMPT
from gehrman.agent.tools import ToolError, dispatch, tool_schemas
from gehrman.config import AppConfig

REQUEST_TIMEOUT_SECONDS = 120.0

# event_type is one of: "tool_call", "tool_result", "tool_error".
EventCallback = Callable[[str, dict[str, Any]], None]


class AgentError(RuntimeError):
    """Raised when the model call fails or the loop hits its step cap."""


def make_client(config: AppConfig) -> OpenAI:
    # Ollama accepts any non-empty string as the key; nothing sensitive is stored.
    return OpenAI(
        base_url=config.agent.base_url,
        api_key="ollama",
        timeout=REQUEST_TIMEOUT_SECONDS,
    )


def start_session() -> list[dict]:
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def run_turn(
    client: OpenAI,
    config: AppConfig,
    messages: list[dict],
    on_event: EventCallback | None = None,
) -> str:
    """Run one user turn: up to max_tool_steps model calls, returns the final answer."""
    tools = tool_schemas()
    for _ in range(config.agent.max_tool_steps):
        try:
            response = client.chat.completions.create(
                model=config.agent.model,
                messages=messages,
                tools=tools,
                temperature=0,
                max_tokens=config.agent.max_tokens_per_message,
            )
        except OpenAIError as exc:
            raise AgentError(f"Model call failed: {exc}") from exc

        message = response.choices[0].message
        entry: dict = {"role": "assistant", "content": message.content or ""}
        if message.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in message.tool_calls
            ]
        messages.append(entry)

        if not message.tool_calls:
            return message.content or ""

        for call in message.tool_calls:
            if on_event is not None:
                on_event(
                    "tool_call",
                    {"name": call.function.name, "arguments": call.function.arguments},
                )
            try:
                result = dispatch(call.function.name, call.function.arguments, config)
                content = json.dumps(result)
                if on_event is not None:
                    on_event("tool_result", {"name": call.function.name, "result": result})
            except ToolError as exc:
                content = json.dumps({"error": str(exc)})
                if on_event is not None:
                    on_event("tool_error", {"name": call.function.name, "error": str(exc)})
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": content}
            )

    raise AgentError(
        f"Exceeded max_tool_steps={config.agent.max_tool_steps} without a final answer"
    )
