"""Tool-use loop against a pluggable model backend.

The loop talks to the model through a `ModelAdapter`. The default
adapter is the OpenAI SDK pointed at Ollama, which preserves the
behaviour the Gehrman MVP shipped with. The thesis evaluation harness
supplies its own adapter (Anthropic or a scripted fake) without the
loop changing shape.

Each turn:

1. Call the model with the running message history plus tool schemas.
2. If it requests tools, dispatch them, record what fired on the input
   side, and feed the results back. Repeat up to `max_tool_steps`.
3. When the model stops calling tools, treat the text as the final
   answer and run the output-side defenses (`scope_check`,
   `number_check`). If any fail, the user-visible answer is replaced
   with a short, redacted stub — the attack canary must never survive
   a failed check.

Callers can ask for a `TurnTrace` to inspect the raw answer, the
delivered answer, which defenses fired, and tool-call history. The
chat command uses the simple string return for its REPL.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from gehrman.agent.adapter import (
    AdapterError,
    AdapterToolCall,
    AdapterUsage,
    ModelAdapter,
    OpenAICompatAdapter,
)
from gehrman.agent.defenses import (
    collect_allowed_from_latest_user_message,
    collect_allowed_from_result,
    number_check,
    scope_check,
)
from gehrman.agent.prompts import SYSTEM_PROMPT
from gehrman.agent.tools import ToolError, dispatch, tool_schemas
from gehrman.config import AppConfig

REQUEST_TIMEOUT_SECONDS = 120.0

# event_type is one of: "tool_call", "tool_result", "tool_error".
EventCallback = Callable[[str, dict[str, Any]], None]


class AgentError(RuntimeError):
    """Raised when the model call fails or the loop hits its step cap."""


@dataclass
class ToolCallTrace:
    name: str
    arguments_json: str
    accepted: bool
    result: Any | None = None
    error: str | None = None
    fired_defenses: list[str] = field(default_factory=list)
    # True when the harness issued the call before the model spoke.
    forced: bool = False


@dataclass
class TurnTrace:
    """Everything the evaluation harness needs to score one turn."""

    raw_answer: str
    final_answer: str
    tool_calls: list[ToolCallTrace] = field(default_factory=list)
    defenses_fired: list[str] = field(default_factory=list)
    usage: AdapterUsage = field(default_factory=AdapterUsage)


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
    client: OpenAI | None,
    config: AppConfig,
    messages: list[dict],
    on_event: EventCallback | None = None,
    *,
    adapter: ModelAdapter | None = None,
    return_trace: bool = False,
    forced_calls: list[tuple[str, dict]] | None = None,
) -> str | TurnTrace:
    """Run one user turn.

    Returns the delivered answer string by default. Pass
    `return_trace=True` to get a `TurnTrace` instead; callers that
    only care about display (the chat REPL) keep the simple return.

    `adapter` overrides the default OpenAI/Ollama adapter.

    `forced_calls` are executed before the first model request, through
    the same dispatch, defense, and allow-list path as a model tool
    call. Omit it and the turn is unchanged.
    """
    if adapter is None:
        if client is None:
            raise AgentError("run_turn requires either `client` or `adapter`.")
        adapter = OpenAICompatAdapter(client, config.agent.model)

    tools = tool_schemas()
    trace = TurnTrace(raw_answer="", final_answer="")
    allowed_hosts: set[str] = set()
    allowed_numbers: set[str] = set()
    # The user is trusted. Tool arguments are not: the model writes them.
    version = config.defenses.check_version
    collect_allowed_from_latest_user_message(
        messages, allowed_hosts, allowed_numbers, version=version
    )

    if forced_calls:
        _perform_tool_calls(
            messages=messages,
            trace=trace,
            config=config,
            calls=[
                AdapterToolCall(
                    id=f"forced-{index}",
                    name=name,
                    arguments_json=json.dumps(arguments),
                )
                for index, (name, arguments) in enumerate(forced_calls)
            ],
            assistant_content="",
            allowed_hosts=allowed_hosts,
            allowed_numbers=allowed_numbers,
            on_event=on_event,
            forced=True,
        )

    for _ in range(config.agent.max_tool_steps):
        try:
            response = adapter.complete(
                messages,
                tools,
                temperature=0.0,
                max_tokens=config.agent.max_tokens_per_message,
            )
        except AdapterError as exc:
            raise AgentError(str(exc)) from exc

        trace.usage.tokens_in += response.usage.tokens_in
        trace.usage.tokens_out += response.usage.tokens_out

        calls = list(response.tool_calls)
        if not calls:
            parsed = _parse_text_tool_call(response.text or "")
            if parsed is not None:
                name, arguments = parsed
                calls = [
                    AdapterToolCall(
                        id="text-0",
                        name=name,
                        arguments_json=json.dumps(arguments),
                    )
                ]
        if not calls:
            messages.append({"role": "assistant", "content": response.text or ""})
            trace.raw_answer = response.text or ""
            trace.final_answer = _apply_output_defenses(
                trace.raw_answer,
                config=config,
                allowed_hosts=allowed_hosts,
                allowed_numbers=allowed_numbers,
                defenses_fired=trace.defenses_fired,
            )
            return trace if return_trace else trace.final_answer

        _perform_tool_calls(
            messages=messages,
            trace=trace,
            config=config,
            calls=calls,
            assistant_content=response.text or "",
            allowed_hosts=allowed_hosts,
            allowed_numbers=allowed_numbers,
            on_event=on_event,
            forced=False,
        )

    raise AgentError(
        f"Exceeded max_tool_steps={config.agent.max_tool_steps} without a final answer"
    )


def _perform_tool_calls(
    *,
    messages: list[dict],
    trace: TurnTrace,
    config: AppConfig,
    calls: list[AdapterToolCall],
    assistant_content: str,
    allowed_hosts: set[str],
    allowed_numbers: set[str],
    on_event: EventCallback | None,
    forced: bool,
) -> None:
    """Dispatch one batch of tool calls and append them to the history."""
    messages.append(
        {
            "role": "assistant",
            "content": assistant_content,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments_json},
                }
                for call in calls
            ],
        }
    )
    for call in calls:
        if on_event is not None:
            on_event("tool_call", {"name": call.name, "arguments": call.arguments_json})
        call_trace = ToolCallTrace(
            name=call.name,
            arguments_json=call.arguments_json,
            accepted=True,
            forced=forced,
        )
        try:
            result, fired = dispatch(call.name, call.arguments_json, config)
            call_trace.result = result
            call_trace.fired_defenses.extend(fired)
            trace.defenses_fired.extend(fired)
            collect_allowed_from_result(
                result, allowed_hosts, allowed_numbers, version=config.defenses.check_version
            )
            content = json.dumps(result)
            if on_event is not None:
                on_event("tool_result", {"name": call.name, "result": result})
        except ToolError as exc:
            call_trace.accepted = False
            call_trace.error = str(exc)
            content = json.dumps({"error": str(exc)})
            if on_event is not None:
                on_event("tool_error", {"name": call.name, "error": str(exc)})
        messages.append({"role": "tool", "tool_call_id": call.id, "content": content})
        trace.tool_calls.append(call_trace)


def _as_tool_call(text: str) -> tuple[str, dict] | None:
    """Parse one JSON object shaped like a tool call, or return None."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    name = data.get("name")
    if not isinstance(name, str) or not name:
        return None
    if "parameters" in data:
        arguments = data["parameters"]
    elif "arguments" in data:
        arguments = data["arguments"]
    else:
        return None
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(arguments, dict):
        return None
    return name, arguments


def _parse_text_tool_call(text: str) -> tuple[str, dict] | None:
    """Recover a tool call written as text instead of a native tool_call.

    Accepts the whole message, or a single fenced block with nothing
    else around it. Prose that merely mentions JSON is left alone.
    """
    parsed = _as_tool_call(text.strip())
    if parsed is not None:
        return parsed
    for match in re.finditer(r"```(?:json)?[^\n]*\n(.*?)```", text, re.DOTALL | re.IGNORECASE):
        parsed = _as_tool_call(match.group(1).strip())
        if parsed is None:
            continue
        outside = (text[: match.start()] + text[match.end() :]).strip()
        if not outside:
            return parsed
    return None


def _apply_output_defenses(
    raw: str,
    *,
    config: AppConfig,
    allowed_hosts: set[str],
    allowed_numbers: set[str],
    defenses_fired: list[str],
) -> str:
    """Run output-side defenses. Mutates `defenses_fired`. Returns the delivered answer.

    If any defense reports `ok=False`, the delivered answer is replaced
    with a short stub so the canary cannot survive. The stub explicitly
    names each failing defense so the trace is self-explanatory.
    """
    reasons: list[str] = []

    ok, evidence = scope_check(
        raw,
        enabled=config.defenses.scope_check,
        allowed_hosts=allowed_hosts,
        version=config.defenses.check_version,
    )
    if config.defenses.scope_check and not ok:
        defenses_fired.append("scope_check")
        reasons.append(f"scope_check: {evidence}")

    ok, evidence = number_check(
        raw,
        enabled=config.defenses.number_check,
        allowed_numbers=allowed_numbers,
        version=config.defenses.check_version,
    )
    if config.defenses.number_check and not ok:
        defenses_fired.append("number_check")
        reasons.append(f"number_check: {evidence}")

    if not reasons:
        return raw
    return "[blocked] " + "; ".join(reasons)
