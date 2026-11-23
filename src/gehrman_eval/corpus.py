"""Attack corpus: YAML-backed cases validated with pydantic.

One attack case has four moving parts:

- `goal`: what the attacker wants (out-of-scope recommendation, fake
  number, re-ranking, tool abuse).
- `form`: how the payload is phrased (explicit instruction, disguised
  instruction, bare false fact).
- `user_query`: the researcher question that will be sent to the agent.
- `carrier`: which program's `policy_excerpt` is poisoned. Either a
  pinned handle, or a rule (`top_hit_for_query`) that selects the
  highest-scored program on the given platform.

The payload is plain text and may contain the literal `{canary}`
placeholder; the injector substitutes it with a per-trial canary at
run time.
"""

from __future__ import annotations

import io
from collections.abc import Iterable
from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from ruamel.yaml import YAML

from gehrman.constants import Platform


class Goal(StrEnum):
    OUT_OF_SCOPE = "out_of_scope"
    FAKE_NUMBER = "fake_number"
    RANKING = "ranking"
    TOOL_ABUSE = "tool_abuse"
    # Adaptive goals: built from real catalog values, so the output
    # checks may let them through by design. See ADAPTIVE_GOALS.
    REAL_OOS_HOST = "real_oos_host"
    CROSS_PROGRAM_HOST = "cross_program_host"
    MISATTRIBUTED_NUMBER = "misattributed_number"
    DEMOTION = "demotion"


# Goals whose payload carries only values that already exist in the
# database (or no checkable value at all). `scope_check` and
# `number_check` compare against real data, so these attacks probe the
# limit of the deterministic layer rather than its by-construction
# coverage. Reports split results by this grouping.
ADAPTIVE_GOALS: frozenset[Goal] = frozenset(
    {Goal.REAL_OOS_HOST, Goal.CROSS_PROGRAM_HOST, Goal.MISATTRIBUTED_NUMBER, Goal.DEMOTION}
)


def goal_group(goal: Goal | str) -> str:
    """`adaptive` for ADAPTIVE_GOALS, `by_construction` otherwise."""
    return "adaptive" if Goal(goal) in ADAPTIVE_GOALS else "by_construction"


# Placeholder each adaptive goal must carry; the injector fills it from
# the database at run time.
REAL_VALUE_PLACEHOLDER: dict[Goal, str] = {
    Goal.REAL_OOS_HOST: "{real_host}",
    Goal.CROSS_PROGRAM_HOST: "{real_host}",
    Goal.MISATTRIBUTED_NUMBER: "{donor_number}",
}


class Form(StrEnum):
    EXPLICIT_INSTRUCTION = "explicit_instruction"
    DISGUISED_INSTRUCTION = "disguised_instruction"
    FALSE_FACT = "false_fact"


class CarrierRule(StrEnum):
    TOP_HIT_FOR_PLATFORM = "top_hit_for_platform"
    # Highest-scored program on the platform that lists at least one
    # out-of-scope hostname. Used by `real_oos_host` attacks.
    TOP_HIT_WITH_OOS_HOST = "top_hit_with_oos_host"


class Carrier(BaseModel):
    """How to pick the program whose policy_excerpt is poisoned.

    Exactly one of `handle` or `rule` must be set. `handle` pins the
    attack to a known program (good for fixtures and reproducibility).
    `rule` lets the injector pick dynamically from whatever database the
    run was pointed at.
    """

    model_config = ConfigDict(extra="forbid")

    platform: Platform
    handle: str | None = None
    rule: CarrierRule | None = None

    @model_validator(mode="after")
    def _one_of_handle_or_rule(self) -> Self:
        if (self.handle is None) == (self.rule is None):
            raise ValueError("Carrier requires exactly one of `handle` or `rule`.")
        return self


class ToolAbuseTarget(BaseModel):
    """What counts as a 'tool abuse attempt' when scoring tool_abuse goals.

    A tool-abuse attack succeeds against the model (model_fooled) when
    the model issues any tool call that matches `tool_name` (e.g.
    `fetch_url`) *or* whose JSON arguments contain `arg_value_fragment`
    (typically the canary host). At least one of the two fields must be
    set — a target with neither can never match.
    """

    model_config = ConfigDict(extra="forbid")

    tool_name: str | None = None
    arg_value_fragment: str | None = None

    @model_validator(mode="after")
    def _at_least_one_signal(self) -> Self:
        if not self.tool_name and not self.arg_value_fragment:
            raise ValueError(
                "ToolAbuseTarget needs tool_name, arg_value_fragment, or both."
            )
        return self


class AttackCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9][a-z0-9-]*$")
    goal: Goal
    form: Form
    user_query: str = Field(min_length=1)
    carrier: Carrier
    payload: str = Field(min_length=1)
    tool_abuse: ToolAbuseTarget | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def _tool_abuse_present_iff_goal(self) -> Self:
        if self.goal == Goal.TOOL_ABUSE and self.tool_abuse is None:
            raise ValueError(f"Attack {self.id}: tool_abuse goal requires a `tool_abuse` target.")
        if self.goal != Goal.TOOL_ABUSE and self.tool_abuse is not None:
            raise ValueError(
                f"Attack {self.id}: `tool_abuse` target only valid when goal == tool_abuse."
            )
        return self

    @model_validator(mode="after")
    def _canary_placeholder_matches_goal(self) -> Self:
        has_placeholder = "{canary}" in self.payload
        if self.goal in {Goal.OUT_OF_SCOPE, Goal.FAKE_NUMBER, Goal.TOOL_ABUSE}:
            if not has_placeholder:
                raise ValueError(
                    f"Attack {self.id}: payload must contain `{{canary}}` for goal {self.goal}."
                )
        # RANKING: canary is optional; the attack succeeds by handle position.
        placeholder = REAL_VALUE_PLACEHOLDER.get(self.goal)
        if placeholder is not None and placeholder not in self.payload:
            raise ValueError(
                f"Attack {self.id}: payload must contain `{placeholder}` for goal {self.goal}."
            )
        if self.goal in ADAPTIVE_GOALS and "{canary}" in self.payload:
            raise ValueError(
                f"Attack {self.id}: adaptive goal {self.goal} must not use `{{canary}}`."
            )
        return self


class AttackCorpus(BaseModel):
    """A loaded, validated attack corpus."""

    model_config = ConfigDict(extra="forbid")

    cases: list[AttackCase]

    @model_validator(mode="after")
    def _unique_ids(self) -> Self:
        seen: set[str] = set()
        for case in self.cases:
            if case.id in seen:
                raise ValueError(f"Duplicate attack id: {case.id}")
            seen.add(case.id)
        return self

    def filter(
        self,
        *,
        ids: Iterable[str] | None = None,
        goals: Iterable[Goal] | None = None,
        forms: Iterable[Form] | None = None,
    ) -> AttackCorpus:
        id_set = set(ids) if ids else None
        goal_set = set(goals) if goals else None
        form_set = set(forms) if forms else None
        selected = [
            case
            for case in self.cases
            if (id_set is None or case.id in id_set)
            and (goal_set is None or case.goal in goal_set)
            and (form_set is None or case.form in form_set)
        ]
        return AttackCorpus(cases=selected)


def load_corpus(path: Path | str) -> AttackCorpus:
    """Read and validate an attack corpus from a YAML file."""
    raw_text = Path(path).read_text(encoding="utf-8")
    return load_corpus_text(raw_text)


def load_corpus_text(text: str) -> AttackCorpus:
    yaml = YAML(typ="safe")
    data = yaml.load(io.StringIO(text))
    if data is None:
        return AttackCorpus(cases=[])
    if isinstance(data, list):
        payload = {"cases": data}
    elif isinstance(data, dict) and "cases" in data:
        payload = data
    else:
        raise ValueError("Corpus YAML must be a list of cases or a mapping with `cases`.")
    return AttackCorpus.model_validate(payload)


class ControlCase(BaseModel):
    """One clean query used to measure false positives."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1, max_length=64, pattern=r"^[a-z0-9][a-z0-9-]*$")
    user_query: str = Field(min_length=1)
    notes: str | None = None


class ControlSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cases: list[ControlCase]

    @model_validator(mode="after")
    def _unique_ids(self) -> Self:
        seen: set[str] = set()
        for case in self.cases:
            if case.id in seen:
                raise ValueError(f"Duplicate control id: {case.id}")
            seen.add(case.id)
        return self


def load_controls(path: Path | str) -> ControlSet:
    raw_text = Path(path).read_text(encoding="utf-8")
    return load_controls_text(raw_text)


def load_controls_text(text: str) -> ControlSet:
    yaml = YAML(typ="safe")
    data = yaml.load(io.StringIO(text))
    if data is None:
        return ControlSet(cases=[])
    if isinstance(data, list):
        payload = {"cases": data}
    elif isinstance(data, dict) and "cases" in data:
        payload = data
    else:
        raise ValueError("Controls YAML must be a list of cases or a mapping with `cases`.")
    return ControlSet.model_validate(payload)
