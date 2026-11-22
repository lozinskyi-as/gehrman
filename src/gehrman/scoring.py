"""Deterministic program fit score, version 2.

The score is a pure function of structured program fields and the
researcher profile. Freshness, added later, also reads Gehrman's own
snapshot history. There is no network call, no randomness, and no
catalog free text: `policy`, target `instruction` / `description`, and
the program `name` are never read. An injected sentence cannot move a
program up or down the ranking.

No public catalog field counts hackers, reports, or participants per
program (checked on the HackerOne, Bugcrowd, and Intigriti snapshots).
Gehrman does not scrape the platforms and stores no API tokens, so the
score has no contributor signal.

A component that cannot be computed is left out, and the remaining
weights are renormalized. A missing value is not treated as zero.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import NamedTuple

from gehrman.catalogs.models import Program
from gehrman.config import ProfileConfig
from gehrman.constants import DEFAULT_WEIGHTS_VERSION, AssetType, Platform, Skill


@dataclass(frozen=True)
class ScoringParams:
    """Every knob the v2 score reads. Experiments pin `weights_version`."""

    weights_version: str
    weights: Mapping[str, float]
    wildcard_breadth_bonus: float
    freshness_window_days: int
    freshness_reference_assets: int
    score_decimals: int
    component_decimals: int


SCORING_PARAMS = ScoringParams(
    weights_version=DEFAULT_WEIGHTS_VERSION,
    weights={
        "skill_fit": 0.45,
        "breadth": 0.30,
        "pays": 0.10,
        "freshness": 0.15,
    },
    wildcard_breadth_bonus=0.15,
    freshness_window_days=30,
    freshness_reference_assets=10,
    score_decimals=1,
    component_decimals=2,
)

# `web` matches web and wildcard assets. `api` matches api assets.
SKILL_ASSET_TYPES: dict[Skill, frozenset[AssetType]] = {
    Skill.WEB: frozenset({AssetType.WEB, AssetType.WILDCARD}),
    Skill.API: frozenset({AssetType.API}),
}

# Built from the public catalog snapshot. Keys are the lowercased
# `asset_type` (HackerOne) or `type` (Bugcrowd, Intigriti) values that
# actually occur. Anything else, including a missing type, is `other`.
ASSET_TYPE_MAP: dict[Platform, dict[str, AssetType]] = {
    Platform.HACKERONE: {
        "url": AssetType.WEB,
        "wildcard": AssetType.WILDCARD,
        "api": AssetType.API,
        "google_play_app_id": AssetType.MOBILE,
        "apple_store_app_id": AssetType.MOBILE,
        "other_apk": AssetType.MOBILE,
        "other_ipa": AssetType.MOBILE,
        "windows_app_store_app_id": AssetType.MOBILE,
        "testflight": AssetType.MOBILE,
        "other": AssetType.OTHER,
        "source_code": AssetType.OTHER,
        "ip_address": AssetType.OTHER,
        "cidr": AssetType.OTHER,
        "hardware": AssetType.OTHER,
        "downloadable_executables": AssetType.OTHER,
        "smart_contract": AssetType.OTHER,
        "ai_model": AssetType.OTHER,
    },
    Platform.BUGCROWD: {
        "website": AssetType.WEB,
        "api": AssetType.API,
        "ios": AssetType.MOBILE,
        "android": AssetType.MOBILE,
        "other": AssetType.OTHER,
        "network": AssetType.OTHER,
        "hardware": AssetType.OTHER,
        "ip_address": AssetType.OTHER,
        "iot": AssetType.OTHER,
        "collection": AssetType.OTHER,
    },
    Platform.INTIGRITI: {
        "url": AssetType.WEB,
        "wildcard": AssetType.WILDCARD,
        "ios": AssetType.MOBILE,
        "android": AssetType.MOBILE,
        "other": AssetType.OTHER,
        "iprange": AssetType.OTHER,
        "device": AssetType.OTHER,
    },
}


class ScoreResult(NamedTuple):
    score: float
    components: dict[str, float]
    extras: dict[str, object]


def canonical_asset_type(platform: Platform, raw: str | None) -> AssetType:
    """Map one catalog type string onto the shared enum."""
    if raw is None:
        return AssetType.OTHER
    key = raw.strip().lower()
    if not key:
        return AssetType.OTHER
    return ASSET_TYPE_MAP.get(platform, {}).get(key, AssetType.OTHER)


def percentile_rank(value: float, population: list[float]) -> float:
    """Fraction of `population` that is less than or equal to `value`.

    Ties share a rank. The maximum value in a non-empty population is 1.
    """
    if not population:
        return 0.0
    return sum(1 for item in population if item <= value) / len(population)


def _round_unit(value: float) -> float:
    clamped = min(1.0, max(0.0, value))
    return round(clamped, SCORING_PARAMS.component_decimals)


def skill_fit(program: Program, profile: ProfileConfig) -> float:
    """Share of in-scope assets whose type matches the profile. Empty scope is 0."""
    assets = program.in_scope_assets
    if not assets:
        return 0.0
    accepted: set[AssetType] = set()
    for skill in profile.skills:
        accepted.update(SKILL_ASSET_TYPES.get(skill, frozenset()))
    matches = sum(1 for asset in assets if asset.type in accepted)
    return matches / len(assets)


def breadth_component(program: Program, scope_logs: list[float]) -> float:
    """Percentile of log1p(scope size), plus a capped wildcard bonus."""
    rank = percentile_rank(math.log1p(program.in_scope_count), scope_logs)
    has_wildcard = any(asset.type is AssetType.WILDCARD for asset in program.in_scope_assets)
    bonus = SCORING_PARAMS.wildcard_breadth_bonus if has_wildcard else 0.0
    return min(1.0, rank + bonus)


def public_number(value: float) -> int | float:
    """Stable JSON number. Whole values stay integers so 5000 is not 5000.0."""
    rounded = round(float(value), SCORING_PARAMS.component_decimals)
    if rounded.is_integer():
        return int(rounded)
    return rounded


# HackerOne time fields are lower-is-better. The unit is undocumented, so
# extras store a within-platform percentile and never the raw value.
_HACKERONE_METRICS: tuple[tuple[str, str, bool], ...] = (
    ("first_response_p", "average_time_to_first_response", True),
    ("report_resolved_p", "average_time_to_report_resolved", True),
    ("bounty_awarded_p", "average_time_to_bounty_awarded", True),
    ("response_efficiency_p", "response_efficiency_percentage", False),
)


def _percentile_points(value: float, population: list[float], *, lower_is_better: bool) -> int:
    if lower_is_better:
        fraction = sum(1 for item in population if item >= value) / len(population)
    else:
        fraction = percentile_rank(value, population)
    return int(round(100 * fraction))


def _hackerone_extras(program: Program, programs: list[Program]) -> dict[str, object]:
    peers = [item for item in programs if item.platform is Platform.HACKERONE]
    extras: dict[str, object] = {}
    for key, attr, lower_is_better in _HACKERONE_METRICS:
        value = getattr(program, attr)
        if value is None:
            continue
        population = [
            float(getattr(item, attr)) for item in peers if getattr(item, attr) is not None
        ]
        if not population:
            continue
        extras[key] = _percentile_points(float(value), population, lower_is_better=lower_is_better)
    if program.in_scope_assets:
        eligible = sum(1 for asset in program.in_scope_assets if asset.bounty_eligible is True)
        extras["bounty_eligible_share"] = public_number(eligible / len(program.in_scope_assets))
    return extras


def _bounty_span(value: float | None, currency: str | None) -> dict[str, object] | None:
    if value is None and currency is None:
        return None
    payload: dict[str, object] = {}
    if value is not None:
        payload["value"] = public_number(value)
    if currency:
        payload["currency"] = currency
    return payload


def platform_extras(program: Program, programs: list[Program]) -> dict[str, object]:
    """Platform-only facts. Displayed, and never an input to the score."""
    if program.platform is Platform.HACKERONE:
        return _hackerone_extras(program, programs)
    if program.platform is Platform.BUGCROWD:
        extras: dict[str, object] = {}
        if program.bounty_max is not None:
            extras["max_payout"] = public_number(program.bounty_max)
        extras["safe_harbor"] = program.safe_harbor
        extras["allows_disclosure"] = program.allows_disclosure
        return extras
    extras = {
        "two_factor_required": program.two_factor_required,
        "tac_required": program.tac_required,
    }
    min_bounty = _bounty_span(program.bounty_min, program.bounty_min_currency)
    max_bounty = _bounty_span(program.bounty_max, program.bounty_max_currency)
    if min_bounty is not None:
        extras["min_bounty"] = min_bounty
    if max_bounty is not None:
        extras["max_bounty"] = max_bounty
    return extras


def format_why(
    components: Mapping[str, float],
    extras: Mapping[str, object],
    platform: str,
) -> str:
    """One compact reason string for `gehrman top` and the agent."""
    parts: list[str] = []
    if "skill_fit" in components:
        parts.append(f"fit {components['skill_fit']:.2f}")
    if "breadth" in components:
        parts.append(f"breadth {components['breadth']:.2f}")
    if platform == Platform.HACKERONE.value:
        for key, label in (
            ("response_efficiency_p", "H1 response"),
            ("first_response_p", "H1 first response"),
        ):
            value = extras.get(key)
            if isinstance(value, int | float):
                parts.append(f"{label} p{int(value)}")
                break
    elif platform == Platform.BUGCROWD.value and isinstance(extras.get("max_payout"), int | float):
        parts.append(f"payout {int(extras['max_payout'])}")
    elif platform == Platform.INTIGRITI.value and isinstance(extras.get("max_bounty"), dict):
        bounty = extras["max_bounty"]
        amount = bounty.get("value") if isinstance(bounty, dict) else None
        if isinstance(amount, int | float):
            parts.append(f"max {int(amount)}")
    new_assets = extras.get("new_assets_30d")
    if isinstance(new_assets, int) and new_assets > 0:
        parts.append(f"new {new_assets}")
    if extras.get("new_program") is True:
        parts.append("new program")
    return " · ".join(parts)


def combine_score(components: Mapping[str, float]) -> float:
    """Weighted mean of the components that are present, scaled to [0, 100]."""
    total_weight = 0.0
    total = 0.0
    for name, value in components.items():
        weight = SCORING_PARAMS.weights[name]
        total_weight += weight
        total += weight * value
    if total_weight == 0:
        return 0.0
    return round(100.0 * total / total_weight, SCORING_PARAMS.score_decimals)


def freshness_component(new_assets_30d: int) -> float:
    """Saturates at 1 once `new_assets_30d` reaches the reference count."""
    reference = SCORING_PARAMS.freshness_reference_assets
    return min(1.0, math.log1p(new_assets_30d) / math.log1p(reference))


def score_programs(
    programs: list[Program],
    profile: ProfileConfig,
    *,
    freshness: Mapping[tuple[str, str], tuple[int, bool]] | None = None,
) -> list[ScoreResult]:
    """Score `programs` against one shared breadth population.

    `freshness` maps `(platform, handle)` to `(new_assets_30d, is_new_program)`.
    Passing None leaves the component out of the mean. Passing a mapping
    makes it available for every program; a missing key is `(0, False)`.
    """
    scope_logs = [math.log1p(program.in_scope_count) for program in programs]
    results: list[ScoreResult] = []
    for program in programs:
        raw: dict[str, float] = {
            "skill_fit": skill_fit(program, profile),
            "breadth": breadth_component(program, scope_logs),
            "pays": 1.0 if program.offers_bounty else 0.0,
        }
        extras = platform_extras(program, programs)
        if freshness is not None:
            key = (program.platform.value, program.handle)
            new_assets, is_new = freshness.get(key, (0, False))
            raw["freshness"] = freshness_component(new_assets)
            extras["new_assets_30d"] = new_assets
            extras["new_program"] = is_new
        components = {name: _round_unit(value) for name, value in raw.items()}
        if profile.bounty_only and not program.offers_bounty:
            score = 0.0
        else:
            score = combine_score(components)
        results.append(ScoreResult(score=score, components=components, extras=extras))
    return results
