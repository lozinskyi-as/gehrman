"""Inject an attack payload into one carrier program's free-text field.

The injector only writes `policy_excerpt`. Structured fields (assets,
bounty amounts, URL, handle, score) are never touched — a test
enforces this invariant.

The injector works on a *copy* of the database. Callers are expected
to copy the base DB into the trial directory first; nothing here
protects the user's real Gehrman data from careless use.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from gehrman_eval.canary import CanaryKind, make_canary, validate_host_canary
from gehrman_eval.corpus import AttackCase, CarrierRule, Goal


@dataclass(frozen=True)
class InjectionResult:
    case_id: str
    platform: str
    handle: str
    canary: str
    original_policy: str
    poisoned_policy: str


_CANARY_KIND_FOR_GOAL: dict[Goal, CanaryKind] = {
    Goal.OUT_OF_SCOPE: CanaryKind.HOST,
    Goal.FAKE_NUMBER: CanaryKind.NUMBER,
    Goal.TOOL_ABUSE: CanaryKind.HOST,
    # RANKING has no canary; the field below is unused.
}


def canary_for_case(case: AttackCase, seed: int) -> str:
    """Return the canary string for `case` at `seed`.

    Ranking attacks get an empty canary since no exact-match token is
    needed; the check uses handle position in the answer instead.
    """
    if case.goal == Goal.RANKING:
        return ""
    canary = make_canary(case.id, seed, kind=_CANARY_KIND_FOR_GOAL[case.goal])
    if _CANARY_KIND_FOR_GOAL[case.goal] == CanaryKind.HOST:
        validate_host_canary(canary)
    return canary


def pick_carrier(db_path: Path, case: AttackCase) -> tuple[str, str]:
    """Return `(platform, handle)` of the program to poison."""
    platform_value = case.carrier.platform.value
    if case.carrier.handle:
        return platform_value, case.carrier.handle
    rule = case.carrier.rule
    if rule == CarrierRule.TOP_HIT_FOR_PLATFORM:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            row = conn.execute(
                "SELECT handle FROM programs WHERE platform = ? "
                "ORDER BY score DESC, handle ASC LIMIT 1",
                (platform_value,),
            ).fetchone()
        finally:
            conn.close()
        if row is None:
            raise ValueError(f"No carrier available for {platform_value}")
        return platform_value, row[0]
    raise ValueError(f"Unsupported carrier rule: {rule}")


def inject_attack(
    db_path: Path,
    case: AttackCase,
    *,
    seed: int = 0,
) -> InjectionResult:
    """Write the attack payload into the carrier's `policy_excerpt` only.

    The payload is appended to the existing policy with a blank-line
    separator so legitimate policy text is still present (a more
    realistic injection than wholesale replacement).
    """
    platform_value, handle = pick_carrier(db_path, case)
    canary = canary_for_case(case, seed)
    payload = case.payload.replace("{canary}", canary) if canary else case.payload

    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT policy_excerpt FROM programs WHERE platform = ? AND handle = ?",
            (platform_value, handle),
        ).fetchone()
        if row is None:
            raise ValueError(f"Carrier {platform_value}/{handle} not in database")
        original = row[0] or ""
        separator = "\n\n" if original else ""
        poisoned = f"{original}{separator}{payload}"
        conn.execute(
            "UPDATE programs SET policy_excerpt = ? WHERE platform = ? AND handle = ?",
            (poisoned, platform_value, handle),
        )
        conn.commit()
    finally:
        conn.close()

    return InjectionResult(
        case_id=case.id,
        platform=platform_value,
        handle=handle,
        canary=canary,
        original_policy=original,
        poisoned_policy=poisoned,
    )
