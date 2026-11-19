"""Report statistics and rendering."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gehrman_eval.report import (
    AttackRecord,
    ControlRecord,
    RunData,
    load_run,
    pick_examples,
    render_csv,
    render_markdown,
    summarize_attacks,
    summarize_controls,
    wilson_interval,
    write_report,
)


def _atk(
    *,
    attack_id: str = "oos-01",
    goal: str = "out_of_scope",
    form: str = "explicit_instruction",
    model: str = "ollama:q",
    profile: str = "none",
    model_fooled: bool = False,
    user_exposed: bool = False,
    defenses_fired: list[str] | None = None,
    error: str | None = None,
) -> AttackRecord:
    return AttackRecord(
        trial_id="t",
        attack_id=attack_id,
        goal=goal,
        form=form,
        model=model,
        profile=profile,
        model_fooled=model_fooled,
        user_exposed=user_exposed,
        defenses_fired=defenses_fired or [],
        user_query="q",
        raw_answer="raw",
        final_answer="final",
        canary="staging-abcd.attacker.example",
        tokens_in=10,
        tokens_out=5,
        latency_s=1.0,
        error=error,
    )


def _ctrl(
    *,
    control_id: str = "ctrl-01",
    model: str = "ollama:q",
    profile: str = "none",
    any_defense_fired: bool = False,
    fired: list[str] | None = None,
) -> ControlRecord:
    return ControlRecord(
        trial_id="t",
        control_id=control_id,
        model=model,
        profile=profile,
        any_defense_fired=any_defense_fired,
        defenses_fired=fired or [],
        user_query="q",
        raw_answer="r",
        final_answer="f",
        latency_s=0.5,
        error=None,
    )


# Wilson intervals


def test_wilson_handles_zero_sample() -> None:
    p = wilson_interval(0, 0)
    assert p.n == 0
    assert p.rate == 0.0
    assert p.lower == 0.0
    assert p.upper == 0.0


def test_wilson_bounds_are_within_zero_one() -> None:
    p = wilson_interval(5, 10)
    assert 0.0 <= p.lower <= p.rate <= p.upper <= 1.0


def test_wilson_all_success_upper_is_one() -> None:
    p = wilson_interval(10, 10)
    assert p.rate == 1.0
    assert p.upper == pytest.approx(1.0, abs=1e-9)
    # Lower is well below 1 for small n
    assert p.lower < 1.0


def test_wilson_all_failure_lower_is_zero() -> None:
    p = wilson_interval(0, 10)
    assert p.rate == 0.0
    assert p.lower == pytest.approx(0.0, abs=1e-9)


def test_wilson_reference_value() -> None:
    # Known reference: 1/10 under 95% Wilson ≈ [0.018, 0.404]
    p = wilson_interval(1, 10)
    assert p.rate == 0.1
    assert p.lower == pytest.approx(0.0179, abs=1e-3)
    assert p.upper == pytest.approx(0.4040, abs=1e-3)


# summarise_attacks / summarise_controls


def test_summarize_attacks_overall() -> None:
    records = [
        _atk(profile="none", model_fooled=True, user_exposed=True),
        _atk(profile="none", model_fooled=False, user_exposed=False),
        _atk(profile="all", model_fooled=True, user_exposed=False),
    ]
    cells = summarize_attacks(records)
    none_cell = next(c for c in cells if c.profile == "none")
    all_cell = next(c for c in cells if c.profile == "all")
    assert none_cell.model_fooled.n == 2
    assert none_cell.model_fooled.successes == 1
    assert all_cell.user_exposed.successes == 0


def test_summarize_attacks_by_form_splits_rows() -> None:
    records = [
        _atk(form="explicit_instruction", model_fooled=True),
        _atk(form="disguised_instruction", model_fooled=False),
    ]
    cells = summarize_attacks(records, slice_by="form")
    forms = {c.slice_value for c in cells}
    assert forms == {"explicit_instruction", "disguised_instruction"}


def test_summarize_controls_counts_fired() -> None:
    records = [
        _ctrl(any_defense_fired=True),
        _ctrl(any_defense_fired=True),
        _ctrl(any_defense_fired=False),
    ]
    cells = summarize_controls(records)
    assert len(cells) == 1
    assert cells[0].any_defense_fired.n == 3
    assert cells[0].any_defense_fired.successes == 2


# Examples


def test_pick_examples_grabs_one_of_each_kind() -> None:
    data = RunData(
        run_dir=Path("/tmp"),
        manifest={},
        attacks=[
            _atk(model_fooled=True, user_exposed=True),
            _atk(model_fooled=True, user_exposed=False, defenses_fired=["scope_check"]),
            _atk(model_fooled=False, user_exposed=False),
        ],
        controls=[_ctrl(any_defense_fired=True, fired=["scope_check"])],
    )
    examples = pick_examples(data)
    titles = [e["title"] for e in examples]
    assert "Attack landed" in titles
    assert "Attack blocked by defenses" in titles
    assert "False positive on clean query" in titles


def test_pick_examples_skips_missing_kinds() -> None:
    data = RunData(
        run_dir=Path("/tmp"),
        manifest={},
        attacks=[_atk(model_fooled=False, user_exposed=False)],
        controls=[],
    )
    assert pick_examples(data) == []


# Rendering


def test_render_markdown_contains_core_sections() -> None:
    data = RunData(
        run_dir=Path("/tmp"),
        manifest={"models": ["ollama:q"], "profiles": ["none"], "repeats": 1},
        attacks=[_atk(model_fooled=True, user_exposed=True)],
        controls=[_ctrl(any_defense_fired=False)],
    )
    md = render_markdown(data)
    assert "Gehrman evaluation report" in md
    assert "Attack success rate per model" in md
    assert "By attack form" in md
    assert "By attack goal" in md
    assert "False-positive rate" in md
    assert "Example traces" in md


def test_render_csv_has_header_and_rows() -> None:
    data = RunData(
        run_dir=Path("/tmp"),
        manifest={},
        attacks=[_atk(model_fooled=True, user_exposed=False)],
        controls=[_ctrl(any_defense_fired=True)],
    )
    csv_text = render_csv(data)
    lines = csv_text.strip().splitlines()
    assert lines[0].startswith("kind,model,profile")
    # At least overall + form + goal + control rows
    assert any(",attack," in ln for ln in lines) or any(ln.startswith("attack,") for ln in lines)
    assert any(ln.startswith("control,") for ln in lines)


# End-to-end: write_report against a minimal run dir


def test_write_report_end_to_end(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "started_at": "2026-01-01T00:00:00Z",
                "models": ["ollama:q"],
                "profiles": ["none", "deterministic"],
                "repeats": 1,
                "corpus_sha256": "abc",
                "base_db_sha256": "def",
            }
        )
    )
    (run_dir / "results.jsonl").write_text(
        json.dumps(
            {
                "trial_id": "t1",
                "attack_id": "oos-01",
                "goal": "out_of_scope",
                "form": "explicit_instruction",
                "model": "ollama:q",
                "profile": "none",
                "repeat": 0,
                "seed": 0,
                "user_query": "q",
                "canary": "staging-abcd.attacker.example",
                "raw_answer": "Try staging-abcd.attacker.example",
                "final_answer": "Try staging-abcd.attacker.example",
                "tool_calls": [],
                "defenses_fired": [],
                "check": {"model_fooled": True, "user_exposed": True, "evidence": "x"},
                "tokens_in": 10,
                "tokens_out": 5,
                "latency_s": 1.0,
                "started_at": "",
                "finished_at": "",
            }
        )
        + "\n"
    )
    md, csv = write_report(run_dir)
    assert md.is_file()
    assert csv.is_file()
    body = md.read_text(encoding="utf-8")
    assert "ollama:q" in body
    assert "out_of_scope" in body


def test_load_run_tolerates_missing_controls(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text("{}")
    (run_dir / "results.jsonl").write_text("")
    data = load_run(run_dir)
    assert data.attacks == []
    assert data.controls == []
