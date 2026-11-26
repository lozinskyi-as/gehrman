"""Suite file + CLI override merge.

A suite YAML pins every parameter so an overnight run is reproducible
from a single file. CLI flags override individual fields; everything
else falls back to the suite defaults, which fall back to the
constants here.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from ruamel.yaml import YAML

from gehrman_eval.corpus import Form, Goal

DEFAULT_MODELS = ("ollama:qwen2.5:3b", "ollama:llama3.2:3b")
# The original four. `deterministic_v2` and `all_v2` are opt-in so a
# default run stays comparable with the v1 defense checks.
DEFAULT_PROFILES = ("none", "prompt", "deterministic", "all")
DEFAULT_REPEATS = 1
DEFAULT_SEED = 0
DEFAULT_TEMPERATURE = 0.0
DEFAULT_CONCURRENCY = 1
DEFAULT_RETRIES = 3
DEFAULT_RETRY_BACKOFF_S = 2.0
DEFAULT_TRIAL_TIMEOUT_S = 0.0


class AttackFilter(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ids: list[str] = Field(default_factory=list)
    goals: list[Goal] = Field(default_factory=list)
    forms: list[Form] = Field(default_factory=list)


class Suite(BaseModel):
    """Resolved, validated run parameters."""

    model_config = ConfigDict(extra="forbid")

    models: list[str] = Field(default_factory=lambda: list(DEFAULT_MODELS))
    profiles: list[str] = Field(default_factory=lambda: list(DEFAULT_PROFILES))
    attacks: AttackFilter = Field(default_factory=AttackFilter)
    repeats: int = Field(default=DEFAULT_REPEATS, ge=1)
    seed: int = DEFAULT_SEED
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0.0, le=2.0)
    concurrency: int = Field(default=DEFAULT_CONCURRENCY, ge=1, le=8)
    retries: int = Field(default=DEFAULT_RETRIES, ge=0)
    retry_backoff_s: float = Field(default=DEFAULT_RETRY_BACKOFF_S, ge=0.0)
    trial_timeout_s: float = Field(default=DEFAULT_TRIAL_TIMEOUT_S, ge=0.0)
    max_trials: int | None = Field(default=None, ge=1)
    corpus: Path = Path("corpus/attacks.yaml")
    controls: Path = Path("corpus/controls.yaml")
    run_controls: bool = True
    out_dir: Path = Path("runs")
    base_db: Path | None = None
    ollama_base_url: str = "http://localhost:11434/v1"
    # `forced` runs get+explain on the carrier before the model speaks.
    # `natural` leaves tool use entirely to the model.
    delivery: Literal["natural", "forced"] = "natural"


def load_suite(path: Path | str | None) -> Suite:
    if path is None:
        return Suite()
    text = Path(path).read_text(encoding="utf-8")
    yaml = YAML(typ="safe")
    data = yaml.load(io.StringIO(text)) or {}
    return Suite.model_validate(data)


def merge_cli_overrides(
    suite: Suite,
    *,
    models: Sequence[str] | None = None,
    profiles: Sequence[str] | None = None,
    attack_ids: Sequence[str] | None = None,
    goals: Sequence[Goal] | None = None,
    forms: Sequence[Form] | None = None,
    repeats: int | None = None,
    seed: int | None = None,
    temperature: float | None = None,
    max_trials: int | None = None,
    corpus: Path | None = None,
    controls: Path | None = None,
    run_controls: bool | None = None,
    out_dir: Path | None = None,
    base_db: Path | None = None,
    ollama_base_url: str | None = None,
    delivery: str | None = None,
) -> Suite:
    """Return a new Suite with the provided CLI flags applied on top."""
    overrides: dict = {}
    if models:
        overrides["models"] = list(models)
    if profiles:
        overrides["profiles"] = list(profiles)
    if attack_ids or goals or forms:
        overrides["attacks"] = AttackFilter(
            ids=list(attack_ids or suite.attacks.ids),
            goals=list(goals or suite.attacks.goals),
            forms=list(forms or suite.attacks.forms),
        )
    if repeats is not None:
        overrides["repeats"] = repeats
    if seed is not None:
        overrides["seed"] = seed
    if temperature is not None:
        overrides["temperature"] = temperature
    if max_trials is not None:
        overrides["max_trials"] = max_trials
    if corpus is not None:
        overrides["corpus"] = corpus
    if controls is not None:
        overrides["controls"] = controls
    if run_controls is not None:
        overrides["run_controls"] = run_controls
    if out_dir is not None:
        overrides["out_dir"] = out_dir
    if base_db is not None:
        overrides["base_db"] = base_db
    if ollama_base_url is not None:
        overrides["ollama_base_url"] = ollama_base_url
    if delivery is not None:
        overrides["delivery"] = delivery
    return suite.model_copy(update=overrides)
