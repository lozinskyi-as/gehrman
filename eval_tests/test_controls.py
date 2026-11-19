"""Control-set schema, loading, and end-to-end runner execution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gehrman_eval.adapters.fake import FakeAdapter, answer
from gehrman_eval.adapters.ollama import OllamaModelInfo
from gehrman_eval.corpus import (
    AttackCorpus,
    ControlCase,
    ControlSet,
    load_controls,
    load_controls_text,
)
from gehrman_eval.runner import Runner, plan_controls
from gehrman_eval.suite import Suite

CONTROLS_PATH = Path(__file__).resolve().parent.parent / "corpus" / "controls.yaml"
MODEL = "fake:scripted"


def test_shipped_controls_load_and_have_at_least_fifteen() -> None:
    controls = load_controls(CONTROLS_PATH)
    assert len(controls.cases) >= 15


def test_duplicate_control_id_rejected() -> None:
    text = """
- id: dup
  user_query: "a"
- id: dup
  user_query: "b"
"""
    with pytest.raises(ValueError, match="Duplicate control id"):
        load_controls_text(text)


def test_bad_control_id_rejected() -> None:
    text = """
- id: "Bad_Id"
  user_query: "x"
"""
    with pytest.raises(ValueError):
        load_controls_text(text)


def test_plan_controls_cartesian_product() -> None:
    controls = ControlSet(
        cases=[ControlCase(id="a", user_query="a"), ControlCase(id="b", user_query="b")]
    )
    suite = Suite(models=[MODEL, "ollama:m2"], profiles=["none", "all"], repeats=2)
    planned = plan_controls(suite, controls)
    assert len(planned) == 2 * 2 * 2 * 2
    assert len({p.trial_id for p in planned}) == len(planned)


def test_runner_executes_controls_and_flags_fired_defenses(
    tmp_path: Path, seed_db
) -> None:
    base_db = seed_db(
        [{"platform": "hackerone", "handle": "shopify", "score": 9.0}]
    )
    corpus = AttackCorpus(cases=[])
    controls = ControlSet(
        cases=[ControlCase(id="ctrl-1", user_query="pick a program")]
    )
    # Clean answer with a host that's NOT in any tool result (no tools called).
    # `scope_check` must fire → this counts as a false positive.
    # One response for precheck + one per control trial.
    fake = FakeAdapter([answer("")] + [answer("Visit shopify.com for details.")])
    suite = Suite(
        models=[MODEL],
        profiles=["deterministic"],
        base_db=base_db,
        out_dir=tmp_path,
    )
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    runner = Runner(suite, run_dir=run_dir)
    runner._adapters[MODEL] = fake
    runner._model_info[MODEL] = OllamaModelInfo(name="fake")
    report = runner.run(corpus, controls=controls)

    assert report.controls_planned == 1
    assert report.controls_executed == 1
    assert report.controls_failed == 0
    line = (run_dir / "controls.jsonl").read_text().strip().splitlines()[0]
    record = json.loads(line)
    assert record["control_id"] == "ctrl-1"
    assert record["any_defense_fired"] is True
    assert "scope_check" in record["defenses_fired"]


def test_runner_skips_controls_when_set_is_none(tmp_path: Path, seed_db) -> None:
    base_db = seed_db([{"platform": "hackerone", "handle": "x", "score": 1.0}])
    fake = FakeAdapter([answer("")])  # one for the precheck
    suite = Suite(
        models=[MODEL],
        profiles=["none"],
        base_db=base_db,
        out_dir=tmp_path,
    )
    run_dir = tmp_path / "run1"
    run_dir.mkdir()
    runner = Runner(suite, run_dir=run_dir)
    runner._adapters[MODEL] = fake
    runner._model_info[MODEL] = OllamaModelInfo(name="fake")
    report = runner.run(AttackCorpus(cases=[]), controls=None)
    assert report.controls_planned == 0
    assert not (run_dir / "controls.jsonl").is_file()
