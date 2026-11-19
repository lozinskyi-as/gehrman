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
   answer and run the output-side defenses (`ai_guard_output`,
   `scope_check`, `number_check`). If any fail, the user-visible
   answer is replaced with a short, redacted stub — the attack
   canary must never survive a failed check.

Callers can ask for a `TurnTrace` to inspect the raw answer, the
delivered answer, which defenses fired, and tool-call history. The
chat command uses the simple string return for its REPL.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from gehrman.agent.adapter import (
    AdapterError,
    AdapterUsage,
    ModelAdapter,
    OpenAICompatAdapter,
)
from gehrman.agent.defenses import (
    Classifier,
    ai_guard_input,
    ai_guard_output,
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
    classifier: Classifier | None = None,
    return_trace: bool = False,
) -> str | TurnTrace:
    """Run one user turn.

    Returns the delivered answer string by default. Pass
    `return_trace=True` to get a `TurnTrace` instead; callers that
    only care about display (the chat REPL) keep the simple return.

    `adapter` overrides the default OpenAI/Ollama adapter. `classifier`
    is a callable that implements the AI-guard verdict; when omitted the
    AI guards no-op (and record so in the trace).
    """
    if adapter is None:
        if client is None:
            raise AgentError("run_turn requires either `client` or `adapter`.")
        adapter = OpenAICompatAdapter(client, config.agent.model)

    tools = tool_schemas()
    trace = TurnTrace(raw_answer="", final_answer="")
    allowed_hosts: set[str] = set()
    allowed_numbers: set[str] = set()

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

        entry: dict = {"role": "assistant", "content": response.text or ""}
        if response.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments_json},
                }
                for call in response.tool_calls
            ]
        messages.append(entry)

        if not response.tool_calls:
            trace.raw_answer = response.text or ""
            trace.final_answer = _apply_output_defenses(
                trace.raw_answer,
                config=config,
                classifier=classifier,
                allowed_hosts=allowed_hosts,
                allowed_numbers=allowed_numbers,
                defenses_fired=trace.defenses_fired,
            )
            return trace if return_trace else trace.final_answer

        for call in response.tool_calls:
            if on_event is not None:
                on_event("tool_call", {"name": call.name, "arguments": call.arguments_json})
            call_trace = ToolCallTrace(
                name=call.name,
                arguments_json=call.arguments_json,
                accepted=True,
            )
            try:
                result, fired = dispatch(call.name, call.arguments_json, config)
                call_trace.result = result
                call_trace.fired_defenses.extend(fired)
                trace.defenses_fired.extend(fired)
                collect_allowed_from_result(result, allowed_hosts, allowed_numbers)
                ok, evidence = ai_guard_input(
                    json.dumps(result),
                    enabled=config.defenses.ai_guard_input,
                    classify=classifier,
                )
                if config.defenses.ai_guard_input and not ok:
                    trace.defenses_fired.append("ai_guard_input")
                    call_trace.fired_defenses.append("ai_guard_input")
                    result = {"error": f"blocked by ai_guard_input: {evidence}"}
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

    raise AgentError(
        f"Exceeded max_tool_steps={config.agent.max_tool_steps} without a final answer"
    )


def _apply_output_defenses(
    raw: str,
    *,
    config: AppConfig,
    classifier: Classifier | None,
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

    ok, evidence = ai_guard_output(
        raw, enabled=config.defenses.ai_guard_output, classify=classifier
    )
    if config.defenses.ai_guard_output and not ok:
        defenses_fired.append("ai_guard_output")
        reasons.append(f"ai_guard_output: {evidence}")

    ok, evidence = scope_check(
        raw, enabled=config.defenses.scope_check, allowed_hosts=allowed_hosts
    )
    if config.defenses.scope_check and not ok:
        defenses_fired.append("scope_check")
        reasons.append(f"scope_check: {evidence}")

    ok, evidence = number_check(
        raw, enabled=config.defenses.number_check, allowed_numbers=allowed_numbers
    )
    if config.defenses.number_check and not ok:
        defenses_fired.append("number_check")
        reasons.append(f"number_check: {evidence}")

    if not reasons:
        return raw
    return "[blocked] " + "; ".join(reasons)
