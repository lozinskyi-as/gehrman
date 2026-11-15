from __future__ import annotations

from gehrman.catalogs.models import (
    BugcrowdProgram,
    HackerOneProgram,
    IntigritiBountyRange,
    IntigritiProgram,
    RawTarget,
    RawTargets,
)
from gehrman.catalogs.normalize import (
    normalize_bugcrowd,
    normalize_hackerone,
    normalize_intigriti,
)
from gehrman.constants import Platform


def test_hackerone_normalization_preserves_structured_fields() -> None:
    entry = HackerOneProgram(
        handle="acme",
        name="Acme",
        url="https://hackerone.com/acme",
        submission_state="open",
        offers_bounties=True,
        policy="Follow the rules. " * 500,
        average_time_to_first_response=12.5,
        targets=RawTargets(
            in_scope=[
                RawTarget(asset_type="URL", asset_identifier="acme.com"),
                RawTarget(asset_type="WILDCARD", asset_identifier="*.acme.com"),
            ],
            out_of_scope=[RawTarget(asset_type="URL", asset_identifier="dev.acme.com")],
        ),
    )
    program = normalize_hackerone(entry, truncate=50)
    assert program.platform is Platform.HACKERONE
    assert program.offers_bounty is True
    assert program.in_scope_count == 2
    assert program.out_of_scope_count == 1
    assert program.asset_types == ["URL", "WILDCARD"]
    assert len(program.policy_excerpt) == 50
    assert program.average_time_to_first_response == 12.5


def test_bugcrowd_normalization_sets_bounty_from_max_payout() -> None:
    entry = BugcrowdProgram(
        name="Beta",
        url="https://bugcrowd.com/beta",
        max_payout=5000.0,
        targets=RawTargets(in_scope=[RawTarget(type="website", uri="beta.example")]),
    )
    program = normalize_bugcrowd(entry, truncate=1000)
    assert program.platform is Platform.BUGCROWD
    assert program.handle == "beta"
    assert program.offers_bounty is True
    assert program.bounty_max == 5000.0
    assert program.asset_types == ["website"]


def test_intigriti_normalization_uses_id_as_handle() -> None:
    entry = IntigritiProgram(
        id="gamma-001",
        name="Gamma",
        url="https://app.intigriti.com/programs/x/gamma/detail",
        min_bounty=IntigritiBountyRange(value=250.0, currency="EUR"),
        max_bounty=IntigritiBountyRange(value=10000.0, currency="EUR"),
        targets=RawTargets(in_scope=[RawTarget(type="api", uri="api.gamma")]),
    )
    program = normalize_intigriti(entry, truncate=1000)
    assert program.platform is Platform.INTIGRITI
    assert program.handle == "gamma-001"
    assert program.offers_bounty is True
    assert program.bounty_min == 250.0
    assert program.bounty_max == 10000.0


def test_bugcrowd_without_payout_is_not_bountied() -> None:
    entry = BugcrowdProgram(
        name="Delta",
        url="https://bugcrowd.com/delta",
        max_payout=None,
        targets=RawTargets(),
    )
    program = normalize_bugcrowd(entry, truncate=1000)
    assert program.offers_bounty is False
    assert program.bounty_max is None
