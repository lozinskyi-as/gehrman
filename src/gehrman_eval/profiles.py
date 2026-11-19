"""Named defense profiles for the ablation.

Each profile is a fixed combination of Gehrman's six defense switches.
The runner maps a profile name to a `DefensesConfig` before building
the per-trial `AppConfig`; nothing else in the harness touches the
booleans directly.
"""

from __future__ import annotations

from gehrman.config import DefensesConfig

PROFILES: dict[str, DefensesConfig] = {
    "none": DefensesConfig(
        sanitize=False,
        data_marking=False,
        ai_guard_input=False,
        ai_guard_output=False,
        scope_check=False,
        number_check=False,
    ),
    "prompt": DefensesConfig(
        sanitize=True,
        data_marking=True,
        ai_guard_input=False,
        ai_guard_output=False,
        scope_check=False,
        number_check=False,
    ),
    "guard": DefensesConfig(
        sanitize=True,
        data_marking=True,
        ai_guard_input=True,
        ai_guard_output=True,
        scope_check=False,
        number_check=False,
    ),
    "deterministic": DefensesConfig(
        sanitize=True,
        data_marking=False,
        ai_guard_input=False,
        ai_guard_output=False,
        scope_check=True,
        number_check=True,
    ),
    "all": DefensesConfig(
        sanitize=True,
        data_marking=True,
        ai_guard_input=True,
        ai_guard_output=True,
        scope_check=True,
        number_check=True,
    ),
}


def get_profile(name: str) -> DefensesConfig:
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown profile {name!r}; available: {sorted(PROFILES)}") from exc
