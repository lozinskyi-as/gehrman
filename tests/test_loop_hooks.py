"""Loop integration tests using a scripted fake adapter.

These cover the M1 hooks: adapter injection, `TurnTrace`, the live
output-defense pipeline, and the fired-defense log. They run entirely
in-process with no network — the fake adapter and a small SQLite
seed replace both the model and the catalog.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from gehrman.agent.adapter import AdapterResponse, AdapterToolCall, AdapterUsage
from gehrman.agent.defenses import TRUSTED_ALLOWLIST_KEYS
from gehrman.agent.loop import TurnTrace, run_turn, start_session
from gehrman.agent.tools import dispatch
from gehrman.catalogs.store import connect
from gehrman.config import AgentConfig, AppConfig, DefensesConfig, ProfileConfig
from gehrman.constants import Skill


class FakeAdapter:
    """Return a scripted list of AdapterResponse objects in order."""

    def __init__(self, scripted: list[AdapterResponse]) -> None:
        self._scripted = list(scripted)
        self.calls: list[dict] = []

    def complete(self, messages, tools, *, temperature, max_tokens) -> AdapterResponse:
        self.calls.append(
            {
                "messages": list(messages),
                "tool_count": len(tools),
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        if not self._scripted:
            raise AssertionError("FakeAdapter exhausted its script")
        return self._scripted.pop(0)


def _answer(text: str, tokens_in: int = 10, tokens_out: int = 5) -> AdapterResponse:
    return AdapterResponse(
        text=text,
        tool_calls=[],
        usage=AdapterUsage(tokens_in=tokens_in, tokens_out=tokens_out),
    )


def _tool_call(name: str, arguments: dict, call_id: str = "call-1") -> AdapterResponse:
    return AdapterResponse(
        text="",
        tool_calls=[AdapterToolCall(id=call_id, name=name, arguments_json=json.dumps(arguments))],
        usage=AdapterUsage(tokens_in=10, tokens_out=0),
    )


def _seed_program(handle: str, policy: str, *, asset_url: str = "https://shopify.com") -> None:
    with connect() as conn:
        conn.execute("DELETE FROM programs")
        conn.execute(
            """
            INSERT INTO programs (
              platform, handle, name, url, offers_bounty,
              bounty_min, bounty_max, asset_types,
              in_scope_count, out_of_scope_count, submission_state,
              average_time_to_first_response, average_time_to_bounty_awarded,
              average_time_to_report_resolved, policy_excerpt, score,
              fetched_at, snapshot_sha256, snapshot_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "hackerone",
                handle,
                "Shopify",
                asset_url,
                1,
                None,
                50000.0,
                json.dumps(["URL"]),
                1,
                0,
                None,
                None,
                None,
                None,
                policy,
                9.0,
                datetime.now(UTC).isoformat(),
                "deadbeef",
                "/tmp/snap.json",
            ),
        )


def _config(**defense_overrides: bool) -> AppConfig:
    defaults = {
        "sanitize": False,
        "data_marking": False,
        "scope_check": False,
        "number_check": False,
    }
    defaults.update(defense_overrides)
    return AppConfig(
        profile=ProfileConfig(skills=[Skill.WEB], bounty_only=False),
        agent=AgentConfig(),
        defenses=DefensesConfig(**defaults),
    )


def test_run_turn_returns_string_by_default(isolated_dirs: dict[str, Path]) -> None:
    _seed_program("shopify", "clean policy text")
    adapter = FakeAdapter([_answer("Try hackerone/shopify.")])
    messages = start_session()
    messages.append({"role": "user", "content": "recommend a program"})
    result = run_turn(client=None, config=_config(), messages=messages, adapter=adapter)
    assert isinstance(result, str)
    assert result == "Try hackerone/shopify."


def test_run_turn_returns_trace_when_requested(isolated_dirs: dict[str, Path]) -> None:
    _seed_program("shopify", "clean policy")
    adapter = FakeAdapter([_answer("Try hackerone/shopify.", tokens_in=7, tokens_out=3)])
    messages = start_session()
    messages.append({"role": "user", "content": "recommend"})
    trace = run_turn(
        client=None, config=_config(), messages=messages, adapter=adapter, return_trace=True
    )
    assert isinstance(trace, TurnTrace)
    assert trace.raw_answer == "Try hackerone/shopify."
    assert trace.final_answer == "Try hackerone/shopify."
    assert trace.usage.tokens_in == 7
    assert trace.usage.tokens_out == 3
    assert trace.defenses_fired == []


def test_input_defenses_fire_and_are_recorded(isolated_dirs: dict[str, Path]) -> None:
    _seed_program("shopify", "policy with a \x00 control byte")
    adapter = FakeAdapter(
        [
            _tool_call("explain", {"platform": "hackerone", "handle": "shopify"}),
            _answer("Got it."),
        ]
    )
    messages = start_session()
    messages.append({"role": "user", "content": "explain shopify"})
    trace = run_turn(
        client=None,
        config=_config(sanitize=True, data_marking=True),
        messages=messages,
        adapter=adapter,
        return_trace=True,
    )
    assert trace.final_answer == "Got it."
    assert "sanitize" in trace.defenses_fired
    assert "data_marking" in trace.defenses_fired
    assert trace.tool_calls[0].name == "explain"
    assert trace.tool_calls[0].accepted is True
    assert "sanitize" in trace.tool_calls[0].fired_defenses


def test_scope_check_blocks_unknown_host_in_answer(isolated_dirs: dict[str, Path]) -> None:
    _seed_program("shopify", "safe policy", asset_url="https://shopify.com")
    adapter = FakeAdapter(
        [
            _tool_call("get", {"platform": "hackerone", "handle": "shopify"}),
            _answer("Attack staging-k7q2.attacker.example right now."),
        ]
    )
    messages = start_session()
    messages.append({"role": "user", "content": "x"})
    trace = run_turn(
        client=None,
        config=_config(scope_check=True),
        messages=messages,
        adapter=adapter,
        return_trace=True,
    )
    assert trace.raw_answer == "Attack staging-k7q2.attacker.example right now."
    assert trace.final_answer.startswith("[blocked]")
    assert "attacker.example" not in trace.final_answer
    assert "scope_check" in trace.defenses_fired


def test_scope_check_allows_hosts_from_tool_result(isolated_dirs: dict[str, Path]) -> None:
    _seed_program("shopify", "safe policy", asset_url="https://shopify.com")
    adapter = FakeAdapter(
        [
            _tool_call("get", {"platform": "hackerone", "handle": "shopify"}),
            _answer("Try shopify.com for bounties."),
        ]
    )
    messages = start_session()
    messages.append({"role": "user", "content": "x"})
    trace = run_turn(
        client=None,
        config=_config(scope_check=True),
        messages=messages,
        adapter=adapter,
        return_trace=True,
    )
    assert trace.final_answer == "Try shopify.com for bounties."
    assert "scope_check" not in trace.defenses_fired


def test_number_check_blocks_unsourced_number(isolated_dirs: dict[str, Path]) -> None:
    _seed_program("shopify", "safe", asset_url="https://shopify.com")
    adapter = FakeAdapter(
        [
            _tool_call("get", {"platform": "hackerone", "handle": "shopify"}),
            _answer("The secret bounty is 99999 USD."),
        ]
    )
    messages = start_session()
    messages.append({"role": "user", "content": "x"})
    trace = run_turn(
        client=None,
        config=_config(number_check=True),
        messages=messages,
        adapter=adapter,
        return_trace=True,
    )
    assert trace.final_answer.startswith("[blocked]")
    assert "99999" not in trace.final_answer
    assert "number_check" in trace.defenses_fired


def test_scope_check_blocks_host_planted_in_policy(
    isolated_dirs: dict[str, Path],
) -> None:
    canary = "staging-abcd.attacker.example"
    _seed_program("shopify", f"API mirrored at {canary}.", asset_url="https://shopify.com")
    adapter = FakeAdapter(
        [
            _tool_call("explain", {"platform": "hackerone", "handle": "shopify"}),
            _answer(f"You can also test {canary}."),
        ]
    )
    messages = start_session()
    messages.append({"role": "user", "content": "explain the policy"})
    trace = run_turn(
        client=None,
        config=_config(scope_check=True),
        messages=messages,
        adapter=adapter,
        return_trace=True,
    )
    assert canary in trace.raw_answer
    assert trace.final_answer.startswith("[blocked]")
    assert canary not in trace.final_answer
    assert "scope_check" in trace.defenses_fired


def test_number_check_blocks_number_planted_in_policy(
    isolated_dirs: dict[str, Path],
) -> None:
    canary = "42424242"
    _seed_program("shopify", f"Critical payout is {canary}.", asset_url="https://shopify.com")
    adapter = FakeAdapter(
        [
            _tool_call("explain", {"platform": "hackerone", "handle": "shopify"}),
            _answer(f"The payout is {canary}."),
        ]
    )
    messages = start_session()
    messages.append({"role": "user", "content": "what is the payout"})
    trace = run_turn(
        client=None,
        config=_config(number_check=True),
        messages=messages,
        adapter=adapter,
        return_trace=True,
    )
    assert canary in trace.raw_answer
    assert trace.final_answer.startswith("[blocked]")
    assert canary not in trace.final_answer
    assert "number_check" in trace.defenses_fired


def test_tool_arguments_do_not_seed_the_allow_list(
    isolated_dirs: dict[str, Path],
) -> None:
    canary = "staging-abcd.attacker.example"
    _seed_program("shopify", "clean policy", asset_url="https://shopify.com")
    adapter = FakeAdapter(
        [
            _tool_call(
                "check_scope",
                {"platform": "hackerone", "handle": "shopify", "candidate": canary},
            ),
            _answer(f"Yes, {canary} is in scope."),
        ]
    )
    messages = start_session()
    messages.append({"role": "user", "content": "check that target"})
    trace = run_turn(
        client=None,
        config=_config(scope_check=True),
        messages=messages,
        adapter=adapter,
        return_trace=True,
    )
    assert canary in trace.raw_answer
    assert trace.final_answer.startswith("[blocked]")
    assert "scope_check" in trace.defenses_fired


def test_structured_fields_and_user_message_are_allowed(
    isolated_dirs: dict[str, Path],
) -> None:
    _seed_program("shopify", "clean policy", asset_url="https://shopify.com")
    answer = "Try shopify.com. Bounty 50000, threshold 7500."
    adapter = FakeAdapter(
        [
            _tool_call("get", {"platform": "hackerone", "handle": "shopify"}),
            _answer(answer),
        ]
    )
    messages = start_session()
    messages.append({"role": "user", "content": "programs with bounties above $7500"})
    trace = run_turn(
        client=None,
        config=_config(scope_check=True, number_check=True),
        messages=messages,
        adapter=adapter,
        return_trace=True,
    )
    assert trace.final_answer == answer
    assert "scope_check" not in trace.defenses_fired
    assert "number_check" not in trace.defenses_fired


def test_run_turn_without_adapter_or_client_raises(isolated_dirs: dict[str, Path]) -> None:
    with pytest.raises(Exception, match="client.*adapter"):
        run_turn(client=None, config=_config(), messages=start_session())


def test_without_forced_calls_the_model_sees_only_the_user_turn(
    isolated_dirs: dict[str, Path],
) -> None:
    del isolated_dirs
    adapter = FakeAdapter([_answer("ok")])
    messages = start_session()
    messages.append({"role": "user", "content": "recommend"})
    run_turn(client=None, config=_config(), messages=messages, adapter=adapter)
    roles = [message["role"] for message in adapter.calls[0]["messages"]]
    assert roles == ["system", "user"]


def test_forced_call_is_shown_to_the_model_before_it_speaks(
    isolated_dirs: dict[str, Path],
) -> None:
    _seed_program("shopify", "SECRET POLICY")
    adapter = FakeAdapter([_answer("ok")])
    messages = start_session()
    messages.append({"role": "user", "content": "explain shopify"})
    trace = run_turn(
        client=None,
        config=_config(),
        messages=messages,
        adapter=adapter,
        return_trace=True,
        forced_calls=[("explain", {"platform": "hackerone", "handle": "shopify"})],
    )
    assert isinstance(trace, TurnTrace)
    assert trace.tool_calls[0].forced is True
    assert trace.tool_calls[0].name == "explain"
    assert "SECRET POLICY" in trace.tool_calls[0].result["policy_excerpt"]
    seen = adapter.calls[0]["messages"]
    assert seen[-1]["role"] == "tool"
    assert "SECRET POLICY" in seen[-1]["content"]


def test_text_json_tool_call_is_executed(isolated_dirs: dict[str, Path]) -> None:
    _seed_program("shopify", "policy")
    adapter = FakeAdapter(
        [
            _answer('{"name": "search", "parameters": {"limit": 1}}'),
            _answer("done"),
        ]
    )
    messages = start_session()
    messages.append({"role": "user", "content": "find one"})
    trace = run_turn(
        client=None, config=_config(), messages=messages, adapter=adapter, return_trace=True
    )
    assert isinstance(trace, TurnTrace)
    assert trace.tool_calls[0].name == "search"
    assert trace.tool_calls[0].accepted is True
    assert trace.tool_calls[0].forced is False
    assert trace.tool_calls[0].result["count"] == 1
    assert trace.final_answer == "done"


def test_text_that_is_not_a_tool_call_stays_the_answer(isolated_dirs: dict[str, Path]) -> None:
    del isolated_dirs
    adapter = FakeAdapter([_answer("not json, and {not a call}")])
    messages = start_session()
    messages.append({"role": "user", "content": "hi"})
    trace = run_turn(
        client=None, config=_config(), messages=messages, adapter=adapter, return_trace=True
    )
    assert isinstance(trace, TurnTrace)
    assert trace.tool_calls == []
    assert trace.raw_answer == "not json, and {not a call}"


def test_unknown_tool_written_as_json_is_rejected(isolated_dirs: dict[str, Path]) -> None:
    del isolated_dirs
    adapter = FakeAdapter(
        [
            _answer('{"name": "fetch_url", "arguments": {"target": "x"}}'),
            _answer("stopped"),
        ]
    )
    messages = start_session()
    messages.append({"role": "user", "content": "hi"})
    trace = run_turn(
        client=None, config=_config(), messages=messages, adapter=adapter, return_trace=True
    )
    assert isinstance(trace, TurnTrace)
    assert trace.tool_calls[0].name == "fetch_url"
    assert trace.tool_calls[0].accepted is False
    assert trace.tool_calls[0].error is not None
    assert "Unknown tool" in trace.tool_calls[0].error
    assert trace.final_answer == "stopped"


def test_unknown_handle_tells_the_model_to_search_first(isolated_dirs: dict[str, Path]) -> None:
    _seed_program("shopify", "policy")
    assert "error" not in TRUSTED_ALLOWLIST_KEYS
    for name, arguments in (
        ("get", {"platform": "hackerone", "handle": "missing"}),
        ("explain", {"platform": "hackerone", "handle": "missing"}),
        ("check_scope", {"platform": "hackerone", "handle": "missing", "candidate": "a.com"}),
    ):
        result, fired = dispatch(name, arguments, _config())
        assert fired == []
        assert result == {
            "found": False,
            "error": (
                "Unknown handle. Call search first and pass the exact "
                "`handle` value from its results."
            ),
        }
