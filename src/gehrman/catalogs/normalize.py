"""Convert validated per-platform entries into unified `Program` records.

The three platforms disagree on field names and shapes. Normalization
collapses that into one storage-ready schema without silently inventing
data: anything missing upstream stays `None` or `0`.
"""

from __future__ import annotations

from gehrman.catalogs.models import (
    BugcrowdProgram,
    HackerOneProgram,
    IntigritiProgram,
    Program,
    RawTarget,
)
from gehrman.constants import Platform


def _asset_types(targets: list[RawTarget]) -> list[str]:
    seen: set[str] = set()
    for item in targets:
        for candidate in (item.asset_type, item.type):
            if candidate:
                cleaned = str(candidate).strip()
                if cleaned:
                    seen.add(cleaned)
    return sorted(seen)


def _truncate(text: str | None, limit: int) -> str:
    if not text:
        return ""
    stripped = text.strip()
    return stripped[:limit]


def _handle_from_url(url: str, fallback: str) -> str:
    tail = url.rstrip("/").rsplit("/", 1)[-1]
    return tail or fallback


def normalize_hackerone(entry: HackerOneProgram, truncate: int) -> Program:
    return Program(
        platform=Platform.HACKERONE,
        handle=entry.handle,
        name=entry.name,
        url=entry.url,
        offers_bounty=bool(entry.offers_bounties),
        asset_types=_asset_types(entry.targets.in_scope),
        in_scope_count=len(entry.targets.in_scope),
        out_of_scope_count=len(entry.targets.out_of_scope),
        submission_state=entry.submission_state,
        average_time_to_first_response=entry.average_time_to_first_response,
        average_time_to_bounty_awarded=entry.average_time_to_bounty_awarded,
        average_time_to_report_resolved=entry.average_time_to_report_resolved,
        policy_excerpt=_truncate(entry.policy, truncate),
    )


def normalize_bugcrowd(entry: BugcrowdProgram, truncate: int) -> Program:
    del truncate  # Bugcrowd catalog has no policy text.
    offers = bool(entry.max_payout and entry.max_payout > 0)
    return Program(
        platform=Platform.BUGCROWD,
        handle=_handle_from_url(entry.url, entry.name),
        name=entry.name,
        url=entry.url,
        offers_bounty=offers,
        bounty_max=entry.max_payout,
        asset_types=_asset_types(entry.targets.in_scope),
        in_scope_count=len(entry.targets.in_scope),
        out_of_scope_count=len(entry.targets.out_of_scope),
    )


def normalize_intigriti(entry: IntigritiProgram, truncate: int) -> Program:
    del truncate
    bounty_min = entry.min_bounty.value if entry.min_bounty else None
    bounty_max = entry.max_bounty.value if entry.max_bounty else None
    offers = (bounty_max or 0) > 0 or (bounty_min or 0) > 0
    handle = entry.id or _handle_from_url(entry.url, entry.name)
    return Program(
        platform=Platform.INTIGRITI,
        handle=handle,
        name=entry.name,
        url=entry.url,
        offers_bounty=bool(offers),
        bounty_min=bounty_min,
        bounty_max=bounty_max,
        asset_types=_asset_types(entry.targets.in_scope),
        in_scope_count=len(entry.targets.in_scope),
        out_of_scope_count=len(entry.targets.out_of_scope),
    )
