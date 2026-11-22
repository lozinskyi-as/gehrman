from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from gehrman.config import (
    FORBIDDEN_SECRET_KEYS,
    AppConfig,
    DefensesConfig,
    ProfileConfig,
    load_config,
    save_config,
)
from gehrman.constants import Skill
from gehrman.paths import config_path


def _sample_profile() -> ProfileConfig:
    return ProfileConfig(skills=[Skill.WEB, Skill.API], bounty_only=True)


def test_defenses_default_to_enabled() -> None:
    defenses = DefensesConfig()
    assert defenses.sanitize is True
    assert defenses.data_marking is True
    assert defenses.scope_check is True
    assert defenses.number_check is True


def test_unknown_skill_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ProfileConfig(skills=["mobile"])  # type: ignore[list-item]


def test_empty_skills_rejected() -> None:
    with pytest.raises(ValidationError):
        ProfileConfig(skills=[])


def test_duplicate_skills_are_deduped() -> None:
    profile = ProfileConfig(skills=[Skill.WEB, Skill.WEB, Skill.API])
    assert profile.skills == [Skill.WEB, Skill.API]


def test_unknown_config_key_is_rejected() -> None:
    with pytest.raises(ValidationError):
        AppConfig.model_validate(
            {
                "profile": {"skills": ["web"], "bounty_only": False},
                "unexpected": True,
            }
        )


def test_save_load_roundtrip(isolated_dirs: dict[str, Path]) -> None:
    config = AppConfig(profile=_sample_profile())
    written = save_config(config)
    assert written == isolated_dirs["config"] / "config.toml"
    loaded = load_config()
    assert loaded == config


def _collect_keys(payload: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            keys.add(str(key).lower())
            keys.update(_collect_keys(value))
    elif isinstance(payload, list):
        for item in payload:
            keys.update(_collect_keys(item))
    return keys


def test_saved_file_has_no_secret_keys(isolated_dirs: dict[str, Path]) -> None:
    save_config(AppConfig(profile=_sample_profile()))
    path = config_path()
    text = path.read_text(encoding="utf-8")
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    assert _collect_keys(data).isdisjoint(FORBIDDEN_SECRET_KEYS)
    assert "Never store API keys" in text


def test_load_missing_file_raises(isolated_dirs: dict[str, Path]) -> None:
    with pytest.raises(FileNotFoundError, match="gehrman init"):
        load_config()


def test_time_unit_default_is_hours() -> None:
    config = AppConfig(profile=_sample_profile())
    assert config.scoring.time_unit == "hours"


def _write_config(text: str) -> None:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_legacy_ai_guard_keys_are_dropped(isolated_dirs: dict[str, Path], capsys) -> None:
    _write_config(
        """
[profile]
skills = ["web"]
bounty_only = false

[defenses]
sanitize = false
ai_guard_input = true
ai_guard_output = false

[ai_guard]
model = "old:latest"
"""
    )
    config = load_config()
    assert config.defenses.sanitize is False
    assert config.defenses.scope_check is True
    captured = capsys.readouterr()
    assert captured.err.count("Warning:") == 1
    assert "gehrman init" in captured.err


def test_unknown_key_still_rejected_when_legacy_keys_present(
    isolated_dirs: dict[str, Path],
) -> None:
    _write_config(
        """
[profile]
skills = ["web"]
bounty_only = false

[defenses]
ai_guard_input = true
scope_chek = true

[ai_guard]
model = "old:latest"
"""
    )
    with pytest.raises(ValidationError):
        load_config()
