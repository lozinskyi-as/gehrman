"""Repeat collapsing, stability and exact McNemar."""

from __future__ import annotations

import pytest

from gehrman_eval.report import AttackRecord
from gehrman_eval.stats import (
    collapse_repeats,
    mcnemar_exact,
    paired_comparisons,
    repeat_stability,
)


def _r(
    attack: str,
    *,
    profile: str = "none",
    model: str = "m",
    fooled: bool = False,
    exposed: bool = False,
    goal: str = "out_of_scope",
    error: str | None = None,
) -> AttackRecord:
    return AttackRecord(
        trial_id="t",
        attack_id=attack,
        goal=goal,
        form="false_fact",
        model=model,
        profile=profile,
        model_fooled=fooled,
        user_exposed=exposed,
        defenses_fired=[],
        user_query="q",
        raw_answer="",
        final_answer="",
        canary="",
        tokens_in=0,
        tokens_out=0,
        latency_s=0.0,
        error=error,
    )


def test_collapse_uses_majority_and_drops_errors() -> None:
    rows = [
        _r("a", fooled=True),
        _r("a", fooled=True),
        _r("a", fooled=False),
        _r("b", fooled=True, error="timeout"),
        _r("b", fooled=False),
        _r("c", error="timeout"),
    ]
    collapsed = {r.attack_id: r for r in collapse_repeats(rows)}
    assert set(collapsed) == {"a", "b"}
    assert collapsed["a"].model_fooled is True
    assert collapsed["b"].model_fooled is False


def test_collapse_tie_counts_for_attacker() -> None:
    rows = [_r("a", exposed=True), _r("a", exposed=False)]
    assert collapse_repeats(rows)[0].user_exposed is True


def test_stability_counts_discordant_attacks() -> None:
    rows = [
        _r("a", fooled=True),
        _r("a", fooled=False),
        _r("b", fooled=True),
        _r("b", fooled=True),
        _r("c", error="x"),
    ]
    (cell,) = repeat_stability(rows)
    assert cell.attacks_with_repeats == 2
    assert cell.discordant_fooled == 1
    assert cell.failed_trials == 1


@pytest.mark.parametrize(
    ("b", "c", "expected"),
    [
        (0, 0, 1.0),
        (5, 0, 0.0625),
        (6, 0, 0.03125),
        (3, 3, 1.0),
        (10, 2, 0.0386),
    ],
)
def test_mcnemar_exact_reference_values(b: int, c: int, expected: float) -> None:
    assert mcnemar_exact(b, c) == pytest.approx(expected, abs=1e-4)


def test_paired_comparisons_within_and_across_models() -> None:
    rows = []
    for i in range(6):
        rows.append(_r(f"a{i}", profile="none", exposed=True))
        rows.append(_r(f"a{i}", profile="all", exposed=False))
        rows.append(_r(f"a{i}", profile="none", model="m2", exposed=i < 3))
    result = paired_comparisons(rows)
    within = next(r for r in result if r.model_a == r.model_b == "m")
    assert (within.only_a, within.only_b, within.n_pairs) == (6, 0, 6)
    assert within.p_value == pytest.approx(0.03125)
    across = next(r for r in result if r.model_a != r.model_b)
    assert across.profile_a == across.profile_b == "none"
    assert (across.only_a, across.only_b) == (3, 0)


def test_paired_comparisons_filter_by_group() -> None:
    rows = [
        _r("x", profile="none", exposed=True, goal="demotion"),
        _r("x", profile="all", exposed=True, goal="demotion"),
        _r("y", profile="none", exposed=True),
        _r("y", profile="all", exposed=False),
    ]
    adaptive = paired_comparisons(rows, group="adaptive")
    assert adaptive[0].n_pairs == 1
    assert adaptive[0].both == 1
