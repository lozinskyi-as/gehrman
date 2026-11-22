"""Freshness comes from Gehrman's snapshot history, not from the catalogs."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from gehrman.catalogs.models import Program, ScopeAsset
from gehrman.catalogs.store import connect, record_asset_history
from gehrman.config import ProfileConfig
from gehrman.constants import AssetType, Platform, Skill
from gehrman.scoring import SCORING_PARAMS, format_why, freshness_component, score_programs

WINDOW = SCORING_PARAMS.freshness_window_days
T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _program(handle: str, identifiers: list[str]) -> Program:
    assets = [ScopeAsset(identifier=item, type=AssetType.WEB) for item in identifiers]
    return Program(
        platform=Platform.HACKERONE,
        handle=handle,
        name=handle,
        url=f"https://hackerone.com/{handle}",
        offers_bounty=True,
        in_scope_assets=assets,
        in_scope_count=len(assets),
    )


def _record(programs: list[Program], seen_at: datetime) -> dict[tuple[str, str], tuple[int, bool]]:
    with connect() as conn:
        return record_asset_history(conn, programs, seen_at, window_days=WINDOW)


def _first_seen(handle: str, identifier: str) -> str:
    with connect() as conn:
        row = conn.execute(
            """
            SELECT first_seen_at FROM asset_history
            WHERE platform = 'hackerone' AND handle = ? AND identifier = ? AND in_scope = 1
            """,
            (handle, identifier),
        ).fetchone()
    assert row is not None
    return str(row[0])


def test_baseline_marks_assets_and_scores_freshness_as_zero(isolated_dirs: dict[str, Path]) -> None:
    del isolated_dirs
    programs = [_program("alpha", ["a.example"]), _program("beta", ["b.example"])]
    stats = _record(programs, T0)
    assert stats[("hackerone", "alpha")] == (0, False)
    assert stats[("hackerone", "beta")] == (0, False)
    with connect() as conn:
        baselines = conn.execute("SELECT DISTINCT baseline FROM asset_history").fetchall()
    assert baselines == [(1,)]
    profile = ProfileConfig(skills=[Skill.WEB], bounty_only=True)
    scored = score_programs(programs, profile, freshness=stats)
    assert scored[0].components["freshness"] == 0.0
    assert scored[1].components["freshness"] == 0.0
    assert scored[0].score == scored[1].score


def test_second_snapshot_counts_only_added_assets(isolated_dirs: dict[str, Path]) -> None:
    del isolated_dirs
    _record([_program("alpha", ["a.example"])], T0)
    stats = _record(
        [_program("alpha", ["a.example", "b.example", "c.example"])],
        T0 + timedelta(days=1),
    )
    assert stats[("hackerone", "alpha")] == (2, False)
    program = _program("alpha", ["a.example", "b.example", "c.example"])
    result = score_programs(
        [program],
        ProfileConfig(skills=[Skill.WEB], bounty_only=False),
        freshness=stats,
    )[0]
    assert result.components["freshness"] == round(freshness_component(2), 2)
    assert result.extras["new_assets_30d"] == 2
    assert "new 2" in format_why(result.components, result.extras, "hackerone")


def test_asset_older_than_the_window_does_not_count(isolated_dirs: dict[str, Path]) -> None:
    del isolated_dirs
    _record([_program("alpha", ["a.example"])], T0)
    added = _record(
        [_program("alpha", ["a.example", "b.example"])],
        T0 + timedelta(days=1),
    )
    assert added[("hackerone", "alpha")][0] == 1
    later = _record(
        [_program("alpha", ["a.example", "b.example"])],
        T0 + timedelta(days=WINDOW + 10),
    )
    assert later[("hackerone", "alpha")] == (0, False)


def test_reappearing_asset_keeps_its_first_seen_at(isolated_dirs: dict[str, Path]) -> None:
    del isolated_dirs
    _record([_program("alpha", ["a.example", "b.example"])], T0)
    original = _first_seen("alpha", "b.example")
    _record([_program("alpha", ["a.example"])], T0 + timedelta(days=1))
    _record([_program("alpha", ["a.example", "b.example"])], T0 + timedelta(days=2))
    assert _first_seen("alpha", "b.example") == original == T0.isoformat()


def test_program_first_seen_after_baseline_is_new(isolated_dirs: dict[str, Path]) -> None:
    del isolated_dirs
    # A baseline program with no assets must not become "new" when assets arrive.
    _record([_program("alpha", ["a.example"]), _program("empty", [])], T0)
    stats = _record(
        [
            _program("alpha", ["a.example"]),
            _program("empty", ["e1.example", "e2.example"]),
            _program("beta", ["b1.example", "b2.example"]),
        ],
        T0 + timedelta(days=2),
    )
    assert stats[("hackerone", "alpha")] == (0, False)
    assert stats[("hackerone", "empty")] == (2, False)
    assert stats[("hackerone", "beta")] == (2, True)
    beta = _program("beta", ["b1.example", "b2.example"])
    result = score_programs(
        [beta],
        ProfileConfig(skills=[Skill.WEB], bounty_only=False),
        freshness=stats,
    )[0]
    assert result.extras["new_program"] is True
    assert "new program" in format_why(result.components, result.extras, "hackerone")


def test_freshness_saturates_and_is_monotonic() -> None:
    assert freshness_component(0) == 0.0
    assert freshness_component(SCORING_PARAMS.freshness_reference_assets) == 1.0
    assert freshness_component(100) == 1.0
    assert freshness_component(3) >= freshness_component(1)
