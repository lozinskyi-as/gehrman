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


def score_programs(
    programs: list[Program],
    profile: ProfileConfig,
    *,
    freshness: Mapping[tuple[str, str], float] | None = None,
) -> list[ScoreResult]:
    """Score `programs` against one shared breadth population.

    `freshness` maps `(platform, handle)` to a component in [0, 1].
    Passing None leaves freshness out of the mean. Passing a mapping
    makes freshness available for every program; a missing key is 0.
    """
    scope_logs = [math.log1p(program.in_scope_count) for program in programs]
    results: list[ScoreResult] = []
    for program in programs:
        raw: dict[str, float] = {
            "skill_fit": skill_fit(program, profile),
            "breadth": breadth_component(program, scope_logs),
            "pays": 1.0 if program.offers_bounty else 0.0,
        }
        if freshness is not None:
            raw["freshness"] = freshness.get((program.platform.value, program.handle), 0.0)
        components = {name: _round_unit(value) for name, value in raw.items()}
        if profile.bounty_only and not program.offers_bounty:
            score = 0.0
        else:
            score = combine_score(components)
        results.append(ScoreResult(score=score, components=components, extras={}))
    return results
