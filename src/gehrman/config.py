"""Load and save the user configuration file.

The file at ~/.config/gehrman/config.toml holds researcher preferences
and defense switches. It must never contain secrets. Gehrman talks to a
local Ollama endpoint, so no API keys are stored; the secret-key guard
below is kept as defense-in-depth against a future provider that needs
one slipping onto disk here.

Unknown keys are rejected so a typo in a defense switch cannot silently
disable a control during the experiment.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path
from typing import Self

import tomli_w
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from gehrman.constants import (
    DEFAULT_MAX_TOKENS_PER_MESSAGE,
    DEFAULT_MAX_TOOL_STEPS,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_TEXT_TRUNCATE,
    DEFAULT_TIME_UNIT,
    DEFAULT_WEIGHTS_VERSION,
    Skill,
)
from gehrman.paths import config_path, ensure_private_dir

CONFIG_HEADER = (
    "# Gehrman configuration. Never store API keys or secrets in this file.\n"
)

# Reject these keys if they ever appear in the serialized config. Belt
# and suspenders for a future provider that needs auth.
FORBIDDEN_SECRET_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "secret",
        "token",
        "password",
    }
)


class ProfileConfig(BaseModel):
    """Researcher profile used later by opportunity scoring."""

    model_config = ConfigDict(extra="forbid")

    skills: list[Skill] = Field(min_length=1)
    bounty_only: bool = False

    @field_validator("skills")
    @classmethod
    def _unique_skills(cls, value: list[Skill]) -> list[Skill]:
        seen: set[Skill] = set()
        unique: list[Skill] = []
        for skill in value:
            if skill not in seen:
                seen.add(skill)
                unique.append(skill)
        return unique


class AgentConfig(BaseModel):
    """Local Ollama endpoint and loop limits. Temperature is fixed at 0 in the agent."""

    model_config = ConfigDict(extra="forbid")

    base_url: str = DEFAULT_OLLAMA_BASE_URL
    model: str = DEFAULT_OLLAMA_MODEL
    max_tool_steps: int = Field(default=DEFAULT_MAX_TOOL_STEPS, ge=1, le=DEFAULT_MAX_TOOL_STEPS)
    max_tokens_per_message: int = Field(default=DEFAULT_MAX_TOKENS_PER_MESSAGE, ge=256)
    text_truncate: int = Field(default=DEFAULT_TEXT_TRUNCATE, ge=1)


class ScoringConfig(BaseModel):
    """Scoring knobs.

    `time_unit` documents how we *display* HackerOne average_time_* values.
    The unit is undocumented upstream, so we store raw numbers and never
    convert them silently. Bugcrowd and Intigriti catalogs expose payout
    amounts (`max_payout`, `min_bounty` / `max_bounty`), not those time
    fields.
    """

    model_config = ConfigDict(extra="forbid")

    time_unit: str = DEFAULT_TIME_UNIT
    weights_version: str = DEFAULT_WEIGHTS_VERSION


class DefensesConfig(BaseModel):
    """Switchable layered defenses for the prompt-injection experiment.

    Defaults are on (secure). Each flag is independent so an ablation can
    disable one control at a time without touching the others.
    """

    model_config = ConfigDict(extra="forbid")

    sanitize: bool = True
    data_marking: bool = True
    scope_check: bool = True
    number_check: bool = True


class AppConfig(BaseModel):
    """Top-level configuration object persisted as TOML."""

    model_config = ConfigDict(extra="forbid")

    profile: ProfileConfig
    agent: AgentConfig = Field(default_factory=AgentConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    defenses: DefensesConfig = Field(default_factory=DefensesConfig)

    @model_validator(mode="after")
    def _no_secret_fields(self) -> Self:
        _assert_no_secret_keys(self.model_dump(mode="json"))
        return self


def _assert_no_secret_keys(payload: object, prefix: str = "") -> None:
    """Raise if a forbidden key name appears anywhere in the structure."""
    if isinstance(payload, dict):
        for key, value in payload.items():
            lowered = str(key).lower()
            path = f"{prefix}.{key}" if prefix else str(key)
            if lowered in FORBIDDEN_SECRET_KEYS:
                msg = (
                    f"Refusing to handle config key {path!r}: secrets do not belong in config.toml"
                )
                raise ValueError(msg)
            _assert_no_secret_keys(value, path)
    elif isinstance(payload, list):
        for index, item in enumerate(payload):
            _assert_no_secret_keys(item, f"{prefix}[{index}]")


# Keys written by init before the AI guard was removed. Dropped so an
# existing config.toml still loads. Every other unknown key is rejected.
_LEGACY_AI_GUARD_DEFENSE_KEYS = frozenset({"ai_guard_input", "ai_guard_output"})


def _drop_legacy_ai_guard(data: dict) -> bool:
    """Remove retired AI-guard keys. Returns True when anything was dropped."""
    dropped = False
    if "ai_guard" in data:
        data.pop("ai_guard")
        dropped = True
    defenses = data.get("defenses")
    if isinstance(defenses, dict):
        for key in _LEGACY_AI_GUARD_DEFENSE_KEYS:
            if key in defenses:
                defenses.pop(key)
                dropped = True
    return dropped


def load_config(path: Path | None = None) -> AppConfig:
    """Read and validate config.toml. Missing file is a hard error."""
    target = path or config_path()
    if not target.is_file():
        msg = f"Config not found at {target}. Run `gehrman init` first."
        raise FileNotFoundError(msg)
    with target.open("rb") as handle:
        data = tomllib.load(handle)
    if _drop_legacy_ai_guard(data):
        print(
            "Warning: config.toml still contains retired AI guard settings. "
            "Re-run `gehrman init` to rewrite it.",
            file=sys.stderr,
        )
    return AppConfig.model_validate(data)


def save_config(config: AppConfig, path: Path | None = None) -> Path:
    """Validate, reject secret keys, and write config.toml."""
    payload = config.model_dump(mode="json")
    _assert_no_secret_keys(payload)
    target = path or config_path()
    ensure_private_dir(target.parent)
    body = tomli_w.dumps(payload)
    target.write_text(CONFIG_HEADER + body, encoding="utf-8")
    return target
