"""Tool-use precheck + manifest recording."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from gehrman.agent.adapter import AdapterError, AdapterResponse, AdapterToolCall
from gehrman_eval.adapters.ollama import OllamaModelInfo
from gehrman_eval.corpus import AttackCorpus
from gehrman_eval.runner import Runner, _redirect_db, tool_use_precheck
from gehrman_eval.suite import Suite


class _ChainFake:
    """search, then explain with a real handle, then a final answer."""

    def __init__(self, handle: str) -> None:
        self.handle = handle
        self.n = 0

    def complete(self, messages, tools, *, temperature, max_tokens):
        del messages, tools, temperature, max_tokens
        self.n += 1
        if self.n == 1:
            return AdapterResponse(
                text="",
                tool_calls=[AdapterToolCall(id="c1", name="search", arguments_json="{}")],
            )
        if self.n == 2:
            return AdapterResponse(
                text="",
                tool_calls=[
                    AdapterToolCall(
                        id="c2",
                        name="explain",
                        arguments_json=json.dumps({"platform": "hackerone", "handle": self.handle}),
                    )
                ],
            )
        return AdapterResponse(text="done", tool_calls=[])


def _as_gehrman_db(source: Path, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "gehrman.db"
    shutil.copy(source, target)
    return target


class _NoToolFake:
    def complete(self, messages, tools, *, temperature, max_tokens):
        return AdapterResponse(text="plain answer", tool_calls=[])


class _ExplodingFake:
    def complete(self, messages, tools, *, temperature, max_tokens):
        raise AdapterError("unreachable")


def test_precheck_passes_when_explain_uses_a_searched_handle(tmp_path: Path, seed_db) -> None:
    from gehrman_eval.runner import _app_config_for

    base = seed_db([{"platform": "hackerone", "handle": "acme", "score": 2.0}])
    config = _app_config_for(
        profile_name="none",
        ollama_base_url="http://localhost:11434/v1",
        model_spec="ollama:x",
    )
    with _redirect_db(_as_gehrman_db(base, tmp_path / "data")):
        result = tool_use_precheck(_ChainFake("acme"), config)
    assert result.passed is True
    assert result.chain_ok is True
    assert result.tool_calls_issued == 2
    assert result.error is None
    assert result.warning is None


def test_precheck_warns_when_the_handle_did_not_come_from_search(tmp_path: Path, seed_db) -> None:
    from gehrman_eval.runner import _app_config_for

    base = seed_db([{"platform": "hackerone", "handle": "acme", "score": 2.0}])
    config = _app_config_for(
        profile_name="none",
        ollama_base_url="http://localhost:11434/v1",
        model_spec="ollama:x",
    )
    with _redirect_db(_as_gehrman_db(base, tmp_path / "data")):
        result = tool_use_precheck(_ChainFake("${INTIGRITI_HANDLE}"), config)
    assert result.passed is False
    assert result.chain_ok is False
    assert result.warning is not None
    assert "chain" in result.warning.lower()


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


def test_manifest_records_precheck_and_model_versions(tmp_path: Path, seed_db) -> None:
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
    runner._adapters["ollama:x"] = _ChainFake("x")
    runner._model_info["ollama:x"] = OllamaModelInfo(name="x", digest="sha256:abcd")
    runner.run(AttackCorpus(cases=[]))

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["extra"]["model_versions"] == {"ollama:x": "sha256:abcd"}
    assert manifest["extra"]["delivery"] == "natural"
    precheck = manifest["extra"]["tool_use_precheck"][0]
    assert precheck["passed"] is True
    assert precheck["chain_ok"] is True
    assert precheck["warning"] is None
    assert precheck["tool_calls_issued"] == 2
