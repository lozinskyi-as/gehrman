"""Suite YAML loading and CLI override merge."""

from __future__ import annotations

from pathlib import Path

from gehrman_eval.corpus import Form, Goal
from gehrman_eval.suite import Suite, load_suite, merge_cli_overrides


def test_suite_defaults_are_sane() -> None:
    suite = Suite()
    assert suite.models == ["ollama:llama3.1"]
    assert "none" in suite.profiles
    assert "all" in suite.profiles
    assert suite.repeats == 1
    assert suite.concurrency == 1  # Ollama is single-process


def test_load_suite_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "suite.yaml"
    path.write_text(
        """
models: [ollama:llama3.1, ollama:qwen2.5]
profiles: [none, prompt]
repeats: 2
seed: 7
max_trials: 10
""".strip(),
        encoding="utf-8",
    )
    suite = load_suite(path)
    assert suite.models == ["ollama:llama3.1", "ollama:qwen2.5"]
    assert suite.profiles == ["none", "prompt"]
    assert suite.repeats == 2
    assert suite.seed == 7
    assert suite.max_trials == 10


def test_load_suite_with_none_returns_defaults() -> None:
    suite = load_suite(None)
    assert suite == Suite()


def test_merge_cli_overrides_replaces_only_provided_fields() -> None:
    suite = Suite(models=["ollama:a"], profiles=["none"], repeats=1, seed=0)
    merged = merge_cli_overrides(
        suite,
        models=["ollama:b"],
        repeats=3,
    )
    assert merged.models == ["ollama:b"]
    assert merged.profiles == ["none"]  # untouched
    assert merged.repeats == 3


def test_merge_cli_overrides_builds_attack_filter() -> None:
    suite = Suite()
    merged = merge_cli_overrides(
        suite,
        attack_ids=["oos-01"],
        goals=[Goal.OUT_OF_SCOPE],
        forms=[Form.EXPLICIT_INSTRUCTION],
    )
    assert merged.attacks.ids == ["oos-01"]
    assert merged.attacks.goals == [Goal.OUT_OF_SCOPE]
    assert merged.attacks.forms == [Form.EXPLICIT_INSTRUCTION]


def test_merge_cli_overrides_leaves_fields_alone_when_none() -> None:
    suite = Suite(repeats=5)
    merged = merge_cli_overrides(suite)
    assert merged.repeats == 5
