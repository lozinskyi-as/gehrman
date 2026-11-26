"""Trial identity and the per-trial result record.

`Trial` is the plan: one (attack × model × profile × repeat) combination.
`TrialResult` is the record written to `results.jsonl` once the trial
has run — everything the report needs to score and explain the outcome.

Trial IDs are deterministic in `(attack_id, model, profile, repeat,
seed)` so a resumed run picks up exactly where it stopped.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from gehrman_eval.corpus import AttackCase, Form, Goal


def trial_id(*, attack_id: str, model: str, profile: str, repeat: int, seed: int) -> str:
    key = f"{attack_id}|{model}|{profile}|{repeat}|{seed}".encode()
    return hashlib.sha256(key).hexdigest()[:16]


def control_trial_id(*, control_id: str, model: str, profile: str, repeat: int, seed: int) -> str:
    key = f"CTRL|{control_id}|{model}|{profile}|{repeat}|{seed}".encode()
    return hashlib.sha256(key).hexdigest()[:16]


@dataclass(frozen=True)
class Trial:
    trial_id: str
    attack_id: str
    goal: Goal
    form: Form
    model: str
    profile: str
    repeat: int
    seed: int
    temperature: float

    @classmethod
    def build(
        cls,
        case: AttackCase,
        *,
        model: str,
        profile: str,
        repeat: int,
        seed: int,
        temperature: float,
    ) -> Trial:
        return cls(
            trial_id=trial_id(
                attack_id=case.id,
                model=model,
                profile=profile,
                repeat=repeat,
                seed=seed,
            ),
            attack_id=case.id,
            goal=case.goal,
            form=case.form,
            model=model,
            profile=profile,
            repeat=repeat,
            seed=seed,
            temperature=temperature,
        )


@dataclass
class ToolCallRecord:
    name: str
    arguments_json: str
    accepted: bool
    result: Any | None = None
    error: str | None = None
    fired_defenses: list[str] = field(default_factory=list)
    forced: bool = False


@dataclass
class CheckRecord:
    model_fooled: bool
    user_exposed: bool
    evidence: str


@dataclass
class TrialResult:
    trial_id: str
    attack_id: str
    goal: str
    form: str
    model: str
    model_version: str | None
    profile: str
    repeat: int
    seed: int
    user_query: str
    carrier_platform: str
    carrier_handle: str
    canary: str
    raw_answer: str
    final_answer: str
    tool_calls: list[ToolCallRecord]
    defenses_fired: list[str]
    check: CheckRecord
    tokens_in: int
    tokens_out: int
    latency_s: float
    started_at: str
    finished_at: str
    error: str | None = None
    # True when the attack text was in a tool result the model actually saw.
    delivered: bool = False

    def to_json_line(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


@dataclass
class ControlTrialResult:
    """Outcome of one clean-input trial. No attack = no canary; the score
    is whether any defense fired on legitimate content."""

    trial_id: str
    control_id: str
    model: str
    model_version: str | None
    profile: str
    repeat: int
    seed: int
    user_query: str
    raw_answer: str
    final_answer: str
    tool_calls: list[ToolCallRecord]
    defenses_fired: list[str]
    any_defense_fired: bool
    tokens_in: int
    tokens_out: int
    latency_s: float
    started_at: str
    finished_at: str
    error: str | None = None

    def to_json_line(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)
