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
from gehrman.constants import Skill

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
