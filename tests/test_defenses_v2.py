"""Defense check version 2: fewer false positives, same canary blocks.

Version 1 is what the original profiles run. These tests pin both.
"""

from __future__ import annotations

from pathlib import Path

from gehrman.agent.defenses import (
    collect_allowed_from_result,
    number_check,
    scope_check,
)
from gehrman_eval.canary import CanaryKind, make_canary
from gehrman_eval.corpus import Goal, load_corpus
from gehrman_eval.profiles import get_profile

_CORPUS = Path(__file__).resolve().parents[1] / "corpus" / "attacks.yaml"

_HOST_NOISE = "Budget $2,500.00 and 1,000.00, score 84.1, delay 5.0, e.g. vue.js at 10.1.2.3."


def test_v2_profiles_opt_in_without_changing_v1() -> None:
    assert get_profile("all").check_version == 1
    assert get_profile("deterministic").check_version == 1
    assert get_profile("all_v2").check_version == 2
    assert get_profile("deterministic_v2").check_version == 2
    assert get_profile("all_v2").data_marking is True
    assert get_profile("deterministic_v2").data_marking is False


def test_v2_ignores_the_false_hosts_from_the_control_run() -> None:
    ok, _ = scope_check(_HOST_NOISE, enabled=True, allowed_hosts=set(), version=2)
    assert ok is True
    blocked, _ = scope_check(_HOST_NOISE, enabled=True, allowed_hosts=set(), version=1)
    assert blocked is False


def test_v2_still_flags_a_real_unknown_host() -> None:
    ok, evidence = scope_check(
        "Try staging-k7q2.attacker.example.",
        enabled=True,
        allowed_hosts=set(),
        version=2,
    )
    assert ok is False
    assert "attacker.example" not in evidence


def test_v2_allows_percentage_forms_of_fraction_fields() -> None:
    numbers: set[str] = set()
    collect_allowed_from_result(
        {"score_components": {"skill_fit": 0.98, "breadth": 1.0}},
        set(),
        numbers,
        version=2,
    )
    ok, _ = number_check(
        "Fit is 98 percent and breadth is 100.",
        enabled=True,
        allowed_numbers=numbers,
        version=2,
    )
    assert ok is True
    assert "98" in numbers
    assert "100" in numbers
    v1_numbers: set[str] = set()
    collect_allowed_from_result(
        {"score_components": {"skill_fit": 0.98, "breadth": 1.0}},
        set(),
        v1_numbers,
        version=1,
    )
    blocked, _ = number_check(
        "Fit is 98 percent and breadth is 100.",
        enabled=True,
        allowed_numbers=v1_numbers,
        version=1,
    )
    assert blocked is False


def test_v2_allows_numbers_that_appear_in_result_key_names() -> None:
    numbers: set[str] = set()
    collect_allowed_from_result({"new_assets_30d": 0}, set(), numbers, version=2)
    ok, _ = number_check(
        "Nothing new in the last 30 days.",
        enabled=True,
        allowed_numbers=numbers,
        version=2,
    )
    assert ok is True
    v1: set[str] = set()
    collect_allowed_from_result({"new_assets_30d": 0}, set(), v1, version=1)
    blocked, _ = number_check(
        "Nothing new in the last 30 days.",
        enabled=True,
        allowed_numbers=v1,
        version=1,
    )
    assert blocked is False


def test_v2_ignores_list_markers() -> None:
    text = "10. First program\n11) Second program"
    ok, _ = number_check(text, enabled=True, allowed_numbers=set(), version=2)
    assert ok is True
    blocked, _ = number_check(text, enabled=True, allowed_numbers=set(), version=1)
    assert blocked is False


def test_corpus_canaries_stay_blocked_in_both_versions() -> None:
    corpus = load_corpus(_CORPUS)
    hosts = 0
    numbers = 0
    for case in corpus.cases:
        if case.goal in (Goal.OUT_OF_SCOPE, Goal.TOOL_ABUSE):
            host = make_canary(case.id, 0, kind=CanaryKind.HOST)
            hosts += 1
            for version in (1, 2):
                ok, _ = scope_check(
                    f"Visit {host}.", enabled=True, allowed_hosts=set(), version=version
                )
                assert ok is False, case.id
        elif case.goal == Goal.FAKE_NUMBER:
            number = make_canary(case.id, 0, kind=CanaryKind.NUMBER)
            numbers += 1
            for version in (1, 2):
                ok, _ = number_check(
                    f"Pay {number}.", enabled=True, allowed_numbers=set(), version=version
                )
                assert ok is False, case.id
    assert hosts > 0
    assert numbers > 0
