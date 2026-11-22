"""Deterministic program fit score.

Pure function of `(Program, ProfileConfig)`; no network, no randomness,
no catalog free-text. Scores are bounded and reproducible so the thesis
experiment can compare runs across snapshots.

The weights here are intentionally simple: scoring refinements belong
behind `scoring.weights_version` so experiments can pin a version.
"""

from __future__ import annotations

import math

from gehrman.catalogs.models import Program
from gehrman.config import ProfileConfig
from gehrman.constants import AssetType, Platform, Skill

# Built from the public catalog snapshot (Part 0 inventory). Keys are the
# lowercased `asset_type` (HackerOne) or `type` (Bugcrowd, Intigriti) values
# that actually occur. Anything else, including a missing type, is `other`.
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


def canonical_asset_type(platform: Platform, raw: str | None) -> AssetType:
    """Map one catalog type string onto the shared enum."""
    if raw is None:
        return AssetType.OTHER
    key = raw.strip().lower()
    if not key:
        return AssetType.OTHER
    return ASSET_TYPE_MAP.get(platform, {}).get(key, AssetType.OTHER)


SKILL_TAGS: dict[Skill, frozenset[str]] = {
    Skill.WEB: frozenset({"url", "wildcard", "website", "web"}),
    Skill.API: frozenset({"api"}),
}


def _program_tag_set(program: Program) -> set[str]:
    tags: set[str] = set()
    for tag in program.asset_types:
        lowered = tag.lower()
        tags.add(lowered)
        for token in lowered.replace("-", "_").split("_"):
            if token:
                tags.add(token)
    return tags


def score_program(program: Program, profile: ProfileConfig) -> float:
    if profile.bounty_only and not program.offers_bounty:
        return 0.0

    score = 0.0
    if program.offers_bounty:
        score += 1.0

    bounty = program.bounty_max or program.bounty_min
    if bounty and bounty > 0:
        score += min(math.log10(bounty), 5.0)

    tags = _program_tag_set(program)
    for skill in profile.skills:
        if tags & SKILL_TAGS.get(skill, frozenset()):
            score += 2.0

    if program.in_scope_count > 0:
        score += 0.5

    return round(score, 3)
