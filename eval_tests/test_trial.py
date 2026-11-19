"""Trial identity and plan expansion."""

from __future__ import annotations

from gehrman_eval.corpus import (
    AttackCase,
    AttackCorpus,
    Carrier,
    CarrierRule,
    Form,
    Goal,
)
from gehrman_eval.runner import plan_trials
from gehrman_eval.suite import AttackFilter, Suite
from gehrman_eval.trial import Trial, trial_id


def _case(id_: str = "oos-01", goal: Goal = Goal.OUT_OF_SCOPE) -> AttackCase:
    return AttackCase(
        id=id_,
        goal=goal,
        form=Form.EXPLICIT_INSTRUCTION,
        user_query="pick one",
        carrier=Carrier(platform="hackerone", rule=CarrierRule.TOP_HIT_FOR_PLATFORM),
        payload="try {canary}",
    )


def test_trial_id_is_deterministic() -> None:
    a = trial_id(attack_id="oos", model="ollama:a", profile="none", repeat=0, seed=0)
    b = trial_id(attack_id="oos", model="ollama:a", profile="none", repeat=0, seed=0)
    assert a == b


def test_trial_id_changes_with_any_input() -> None:
    base = trial_id(attack_id="oos", model="ollama:a", profile="none", repeat=0, seed=0)
    assert base != trial_id(attack_id="xxx", model="ollama:a", profile="none", repeat=0, seed=0)
    assert base != trial_id(attack_id="oos", model="ollama:b", profile="none", repeat=0, seed=0)
    assert base != trial_id(attack_id="oos", model="ollama:a", profile="all", repeat=0, seed=0)
    assert base != trial_id(attack_id="oos", model="ollama:a", profile="none", repeat=1, seed=0)
    assert base != trial_id(attack_id="oos", model="ollama:a", profile="none", repeat=0, seed=1)


def test_plan_trials_is_cartesian_product() -> None:
    corpus = AttackCorpus(cases=[_case("a"), _case("b")])
    suite = Suite(
        models=["ollama:m1", "ollama:m2"],
        profiles=["none", "all"],
        repeats=2,
        seed=0,
        temperature=0.0,
    )
    planned = plan_trials(suite, corpus)
    assert len(planned) == 2 * 2 * 2 * 2  # models * profiles * cases * repeats
    # All trial IDs are unique.
    ids = {pt.trial.trial_id for pt in planned}
    assert len(ids) == len(planned)


def test_plan_applies_attack_filters() -> None:
    corpus = AttackCorpus(
        cases=[_case("a", goal=Goal.OUT_OF_SCOPE), _case("b", goal=Goal.FAKE_NUMBER)]
    )
    suite = Suite(
        models=["ollama:m1"],
        profiles=["none"],
        attacks=AttackFilter(goals=[Goal.FAKE_NUMBER]),
        repeats=1,
    )
    planned = plan_trials(suite, corpus)
    assert len(planned) == 1
    assert planned[0].case.id == "b"


def test_trial_build_fills_metadata() -> None:
    case = _case("oos-01")
    trial = Trial.build(
        case, model="ollama:m1", profile="none", repeat=0, seed=7, temperature=0.0
    )
    assert trial.attack_id == "oos-01"
    assert trial.goal == Goal.OUT_OF_SCOPE
    assert trial.seed == 7
