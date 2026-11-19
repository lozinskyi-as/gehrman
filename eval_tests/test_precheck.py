"""Tool-use precheck + manifest recording."""

from __future__ import annotations

import json
from pathlib import Path

from gehrman.agent.adapter import AdapterError, AdapterResponse, AdapterToolCall
from gehrman_eval.adapters.ollama import OllamaModelInfo
from gehrman_eval.corpus import AttackCorpus
from gehrman_eval.runner import Runner, tool_use_precheck
from gehrman_eval.suite import Suite


class _ToolCallingFake:
    def complete(self, messages, tools, *, temperature, max_tokens):
        return AdapterResponse(
            text="",
            tool_calls=[
                AdapterToolCall(id="c1", name="search", arguments_json="{}")
            ],
        )


class _NoToolFake:
    def complete(self, messages, tools, *, temperature, max_tokens):
        return AdapterResponse(text="plain answer", tool_calls=[])


class _ExplodingFake:
    def complete(self, messages, tools, *, temperature, max_tokens):
        raise AdapterError("unreachable")


def test_precheck_passes_when_tool_call_issued() -> None:
    from gehrman_eval.runner import _app_config_for

    config = _app_config_for(
        profile_name="none",
        ollama_base_url="http://localhost:11434/v1",
        model_spec="ollama:x",
    )
    result = tool_use_precheck(_ToolCallingFake(), config)
    assert result.passed is True
    assert result.tool_calls_issued == 1
    assert result.error is None


def test_precheck_fails_when_no_tool_call() -> None:
    from gehrman_eval.runner import _app_config_for

    config = _app_config_for(
        profile_name="none",
        ollama_base_url="http://localhost:11434/v1",
        model_spec="ollama:x",
    )
    result = tool_use_precheck(_NoToolFake(), config)
    assert result.passed is False
    assert result.tool_calls_issued == 0


def test_precheck_fails_when_adapter_errors() -> None:
    from gehrman_eval.runner import _app_config_for

    config = _app_config_for(
        profile_name="none",
        ollama_base_url="http://localhost:11434/v1",
        model_spec="ollama:x",
    )
    result = tool_use_precheck(_ExplodingFake(), config)
    assert result.passed is False
    assert result.error is not None
    assert "unreachable" in result.error


def test_manifest_records_precheck_and_model_versions(
    tmp_path: Path, seed_db
) -> None:
    base_db = seed_db([{"platform": "hackerone", "handle": "x", "score": 1.0}])
    suite = Suite(
        models=["ollama:x"],
        profiles=["none"],
        base_db=base_db,
        out_dir=tmp_path,
    )
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    runner = Runner(suite, run_dir=run_dir)
    runner._adapters["ollama:x"] = _ToolCallingFake()
    runner._model_info["ollama:x"] = OllamaModelInfo(name="x", digest="sha256:abcd")
    runner.run(AttackCorpus(cases=[]))

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["extra"]["model_versions"] == {"ollama:x": "sha256:abcd"}
    assert manifest["extra"]["tool_use_precheck"][0]["passed"] is True
    assert manifest["extra"]["tool_use_precheck"][0]["tool_calls_issued"] == 1
