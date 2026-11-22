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
    ScopeAsset,
)
from gehrman.constants import Platform
from gehrman.scoring import canonical_asset_type


def _asset_types(targets: list[RawTarget]) -> list[str]:
    seen: set[str] = set()
    for item in targets:
        for candidate in (item.asset_type, item.type):
            if candidate:
                cleaned = str(candidate).strip()
                if cleaned:
                    seen.add(cleaned)
    return sorted(seen)


def _first_text(*values: str | None) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _raw_type(item: RawTarget) -> str | None:
    return _first_text(item.asset_type, item.type)


def _identifier(platform: Platform, item: RawTarget) -> str | None:
    """Asset identifier field, per platform, from the live catalogs.

    HackerOne uses `asset_identifier`. Bugcrowd always fills `target`
    (and `name`); `uri` is often empty, so it is only a fallback.
    Intigriti uses `endpoint`.
    """
    if platform is Platform.HACKERONE:
        values = (item.asset_identifier,)
    elif platform is Platform.BUGCROWD:
        values = (item.target, item.uri, item.name)
    else:
        values = (item.endpoint,)
    return _first_text(*values)


def _scope_assets(platform: Platform, targets: list[RawTarget]) -> list[ScopeAsset]:
    """Structured scope entries. Instruction and description text are dropped."""
    assets: list[ScopeAsset] = []
    for item in targets:
        identifier = _identifier(platform, item)
        if identifier is None:
            continue
        eligible = item.eligible_for_bounty if platform is Platform.HACKERONE else None
        assets.append(
            ScopeAsset(
                identifier=identifier,
                type=canonical_asset_type(platform, _raw_type(item)),
                bounty_eligible=eligible,
            )
        )
    return assets


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
        in_scope_assets=_scope_assets(Platform.HACKERONE, entry.targets.in_scope),
        out_of_scope_assets=_scope_assets(Platform.HACKERONE, entry.targets.out_of_scope),
        in_scope_count=len(entry.targets.in_scope),
        out_of_scope_count=len(entry.targets.out_of_scope),
        submission_state=entry.submission_state,
        average_time_to_first_response=entry.average_time_to_first_response,
        average_time_to_bounty_awarded=entry.average_time_to_bounty_awarded,
        average_time_to_report_resolved=entry.average_time_to_report_resolved,
        response_efficiency_percentage=entry.response_efficiency_percentage,
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
        in_scope_assets=_scope_assets(Platform.BUGCROWD, entry.targets.in_scope),
        out_of_scope_assets=_scope_assets(Platform.BUGCROWD, entry.targets.out_of_scope),
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
        in_scope_assets=_scope_assets(Platform.INTIGRITI, entry.targets.in_scope),
        out_of_scope_assets=_scope_assets(Platform.INTIGRITI, entry.targets.out_of_scope),
        in_scope_count=len(entry.targets.in_scope),
        out_of_scope_count=len(entry.targets.out_of_scope),
    )
