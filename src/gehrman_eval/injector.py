"""Inject an attack payload into one carrier program's free-text field.

Adaptive goals (see `corpus.ADAPTIVE_GOALS`) do not use a synthetic
canary. Their payload carries a real value read from the same database:
an out-of-scope host of the carrier, an in-scope host of another
program, or another program's maximum bounty. The value is stored in
`InjectionResult.canary` so the checks can look for it.

The injector only writes `policy_excerpt`. Structured fields (assets,
bounty amounts, URL, handle, score) are never touched — a test
enforces this invariant.

The injector works on a *copy* of the database. Callers are expected
to copy the base DB into the trial directory first; nothing here
protects the user's real Gehrman data from careless use.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from gehrman.scope_match import normalize_host
from gehrman_eval.canary import CanaryKind, make_canary, validate_host_canary
from gehrman_eval.corpus import ADAPTIVE_GOALS, AttackCase, CarrierRule, Goal


@dataclass(frozen=True)
class InjectionResult:
    case_id: str
    platform: str
    handle: str
    canary: str
    original_policy: str
    poisoned_policy: str
    # For `misattributed_number`: the program the number really belongs to.
    donor_handle: str = ""


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
    if rule == CarrierRule.TOP_HIT_WITH_OOS_HOST:
        for handle, _in, out in _programs_by_score(db_path, platform_value):
            if _plain_hosts(out):
                return platform_value, handle
        raise ValueError(f"No {platform_value} program lists an out-of-scope host")
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
    donor_handle = ""
    if case.goal in ADAPTIVE_GOALS:
        canary, donor_handle = resolve_real_value(db_path, case, platform_value, handle)
        payload = (
            case.payload.replace("{real_host}", canary).replace("{donor_number}", canary)
        )
    else:
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
        donor_handle=donor_handle,
    )


def _programs_by_score(db_path: Path, platform: str) -> list[tuple[str, list, list]]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute(
            "SELECT handle, in_scope_assets, out_of_scope_assets FROM programs "
            "WHERE platform = ? ORDER BY score DESC, handle ASC",
            (platform,),
        ).fetchall()
    finally:
        conn.close()
    return [(h, _json_list(i), _json_list(o)) for h, i, o in rows]


def _json_list(raw: str | None) -> list:
    try:
        value = json.loads(raw or "[]")
    except json.JSONDecodeError:
        return []
    return value if isinstance(value, list) else []


def _plain_hosts(assets: list) -> list[str]:
    """Hostnames of non-wildcard, non-mobile scope entries, in catalog order."""
    hosts: list[str] = []
    for asset in assets:
        if not isinstance(asset, dict) or asset.get("type") == "mobile":
            continue
        identifier = str(asset.get("identifier") or "")
        if "*" in identifier:
            continue
        host = normalize_host(identifier)
        if host and host not in hosts:
            hosts.append(host)
    return hosts


def resolve_real_value(
    db_path: Path, case: AttackCase, platform: str, carrier: str
) -> tuple[str, str]:
    """Return `(value, donor_handle)` for an adaptive attack. Raises if none exists."""
    if case.goal == Goal.DEMOTION:
        return "", ""
    programs = _programs_by_score(db_path, platform)
    carrier_row = next((p for p in programs if p[0] == carrier), None)
    if carrier_row is None:
        raise ValueError(f"Carrier {platform}/{carrier} not in database")
    if case.goal == Goal.REAL_OOS_HOST:
        hosts = _plain_hosts(carrier_row[2])
        if not hosts:
            raise ValueError(f"Carrier {platform}/{carrier} has no out-of-scope host")
        return hosts[0], ""
    if case.goal == Goal.CROSS_PROGRAM_HOST:
        own = set(_plain_hosts(carrier_row[1]))
        for handle, in_scope, _out in programs:
            if handle == carrier:
                continue
            for host in _plain_hosts(in_scope):
                if host not in own:
                    return host, handle
        raise ValueError(f"No other {platform} program has a plain in-scope host")
    if case.goal == Goal.MISATTRIBUTED_NUMBER:
        return _donor_number(db_path, platform, carrier)
    raise ValueError(f"Goal {case.goal} has no real value to resolve")


def _donor_number(db_path: Path, platform: str, carrier: str) -> tuple[str, str]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        own = conn.execute(
            "SELECT bounty_max FROM programs WHERE platform = ? AND handle = ?",
            (platform, carrier),
        ).fetchone()
        rows = conn.execute(
            "SELECT handle, bounty_max FROM programs WHERE platform = ? AND handle != ? "
            "AND bounty_max IS NOT NULL AND bounty_max >= 100 "
            "ORDER BY score DESC, handle ASC",
            (platform, carrier),
        ).fetchall()
    finally:
        conn.close()
    own_value = own[0] if own else None
    for handle, value in rows:
        if own_value is not None and float(value) == float(own_value):
            continue
        number = int(value) if float(value).is_integer() else value
        return str(number), handle
    raise ValueError(f"No donor bounty for {platform}/{carrier}")
