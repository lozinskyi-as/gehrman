"""Host and number checks ignore catalog noise and still block canaries."""

from __future__ import annotations

from pathlib import Path

from gehrman.agent.defenses import (
    collect_allowed_from_result,
    number_check,
    scope_check,
)
from gehrman_eval.canary import CanaryKind, make_canary
from gehrman_eval.corpus import Goal, load_corpus
from gehrman_eval.profiles import PROFILES, get_profile

_CORPUS = Path(__file__).resolve().parents[1] / "corpus" / "attacks.yaml"

_HOST_NOISE = "Budget $2,500.00 and 1,000.00, score 84.1, delay 5.0, e.g. vue.js at 10.1.2.3."


def test_profiles_are_the_four_switches() -> None:
    assert sorted(PROFILES) == ["all", "deterministic", "none", "prompt"]
    assert get_profile("all").data_marking is True
    assert get_profile("deterministic").data_marking is False
    assert get_profile("deterministic").scope_check is True
    assert get_profile("deterministic").number_check is True


def test_ignores_the_false_hosts_from_the_control_run() -> None:
    ok, _ = scope_check(_HOST_NOISE, enabled=True, allowed_hosts=set())
    assert ok is True


def test_still_flags_a_real_unknown_host() -> None:
    ok, evidence = scope_check(
        "Try staging-k7q2.attacker.example.",
        enabled=True,
        allowed_hosts=set(),
    )
    assert ok is False
    assert "attacker.example" not in evidence


def test_allows_percentage_forms_of_fraction_fields() -> None:
    numbers: set[str] = set()
    collect_allowed_from_result(
        {"score_components": {"skill_fit": 0.98, "breadth": 1.0}},
        set(),
        numbers,
    )
    ok, _ = number_check(
        "Fit is 98 percent and breadth is 100.",
        enabled=True,
        allowed_numbers=numbers,
    )
    assert ok is True
    assert "98" in numbers
    assert "100" in numbers


def test_allows_numbers_that_appear_in_result_key_names() -> None:
    numbers: set[str] = set()
    collect_allowed_from_result({"new_assets_30d": 0}, set(), numbers)
    ok, _ = number_check(
        "Nothing new in the last 30 days.",
        enabled=True,
        allowed_numbers=numbers,
    )
    assert ok is True


def test_ignores_list_markers() -> None:
    text = "10. First program\n11) Second program"
    ok, _ = number_check(text, enabled=True, allowed_numbers=set())
    assert ok is True


def test_corpus_canaries_stay_blocked() -> None:
    corpus = load_corpus(_CORPUS)
    hosts = 0
    numbers = 0
    for case in corpus.cases:
        if case.goal in (Goal.OUT_OF_SCOPE, Goal.TOOL_ABUSE):
            host = make_canary(case.id, 0, kind=CanaryKind.HOST)
            hosts += 1
            ok, _ = scope_check(f"Visit {host}.", enabled=True, allowed_hosts=set())
            assert ok is False, case.id
        elif case.goal == Goal.FAKE_NUMBER:
            number = make_canary(case.id, 0, kind=CanaryKind.NUMBER)
            numbers += 1
            ok, _ = number_check(f"Pay {number}.", enabled=True, allowed_numbers=set())
            assert ok is False, case.id
    assert hosts > 0
    assert numbers > 0
