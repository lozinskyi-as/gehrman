from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from typer.testing import CliRunner

from gehrman.catalogs.store import connect, db_path
from gehrman.cli import app

runner = CliRunner()


def _insert(rows: list[tuple]) -> None:
    with connect() as conn:
        conn.execute("DELETE FROM programs")
        conn.executemany(
            """
            INSERT INTO programs (
              platform, handle, name, url, offers_bounty,
              bounty_min, bounty_max, asset_types,
              in_scope_count, out_of_scope_count, submission_state,
              average_time_to_first_response, average_time_to_bounty_awarded,
              average_time_to_report_resolved, policy_excerpt, score,
              fetched_at, snapshot_sha256, snapshot_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def _row(
    *,
    platform: str,
    handle: str,
    name: str,
    bounty_max: float | None,
    score: float,
    assets: list[str],
) -> tuple:
    return (
        platform,
        handle,
        name,
        f"https://example.test/{handle}",
        1 if bounty_max else 0,
        None,
        bounty_max,
        json.dumps(assets),
        1,
        0,
        None,
        None,
        None,
        None,
        "",
        score,
        datetime.now(UTC).isoformat(),
        "deadbeef",
        "/tmp/snap.json",
    )


def test_top_errors_without_database(isolated_dirs: dict[str, Path]) -> None:
    result = runner.invoke(app, ["top"])
    assert result.exit_code == 1
    # Rich wraps the path, so the two words may land on different lines.
    assert "gehrman update" in result.output.replace("\n", "")


def test_top_lists_highest_scores_first(isolated_dirs: dict[str, Path]) -> None:
    _insert(
        [
            _row(
                platform="hackerone",
                handle="low",
                name="Low",
                bounty_max=100,
                score=1.5,
                assets=["URL"],
            ),
            _row(
                platform="bugcrowd",
                handle="high",
                name="High",
                bounty_max=10000,
                score=9.0,
                assets=["website"],
            ),
            _row(
                platform="intigriti",
                handle="mid",
                name="Mid",
                bounty_max=1000,
                score=4.0,
                assets=["api"],
            ),
        ]
    )
    result = runner.invoke(app, ["top", "--limit", "2"])
    assert result.exit_code == 0, result.output
    first_high = result.output.index("High")
    first_mid = result.output.index("Mid")
    assert first_high < first_mid
    assert "Low" not in result.output


def test_top_filters_by_platform(isolated_dirs: dict[str, Path]) -> None:
    _insert(
        [
            _row(
                platform="hackerone",
                handle="h1",
                name="One",
                bounty_max=5000,
                score=6.0,
                assets=["URL"],
            ),
            _row(
                platform="bugcrowd",
                handle="b1",
                name="Beta",
                bounty_max=5000,
                score=6.0,
                assets=["website"],
            ),
        ]
    )
    result = runner.invoke(app, ["top", "--platform", "bugcrowd"])
    assert result.exit_code == 0, result.output
    assert "Beta" in result.output
    assert "One" not in result.output


def test_top_filters_by_min_bounty(isolated_dirs: dict[str, Path]) -> None:
    _insert(
        [
            _row(
                platform="hackerone",
                handle="cheap",
                name="Cheap",
                bounty_max=100,
                score=2.0,
                assets=["URL"],
            ),
            _row(
                platform="hackerone",
                handle="rich",
                name="Rich",
                bounty_max=20000,
                score=8.0,
                assets=["URL"],
            ),
        ]
    )
    result = runner.invoke(app, ["top", "--min-bounty", "1000"])
    assert result.exit_code == 0, result.output
    assert "Rich" in result.output
    assert "Cheap" not in result.output


def test_top_excludes_zero_scores(isolated_dirs: dict[str, Path]) -> None:
    _insert(
        [
            _row(
                platform="hackerone",
                handle="z",
                name="Zero",
                bounty_max=None,
                score=0.0,
                assets=["URL"],
            ),
            _row(
                platform="hackerone",
                handle="p",
                name="Positive",
                bounty_max=500,
                score=3.0,
                assets=["URL"],
            ),
        ]
    )
    result = runner.invoke(app, ["top"])
    assert result.exit_code == 0, result.output
    assert "Positive" in result.output
    assert "Zero" not in result.output


def test_top_escapes_rich_markup_in_names(isolated_dirs: dict[str, Path]) -> None:
    _insert(
        [
            _row(
                platform="hackerone",
                handle="evil",
                name="Acme [red]PWN[/red]",
                bounty_max=1000,
                score=5.0,
                assets=["URL"],
            ),
        ]
    )
    # Give Rich enough width so the literal brackets aren't wrapped away.
    result = runner.invoke(app, ["top"], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    collapsed = "".join(result.output.split())
    assert "[red]PWN[/red]" in collapsed


def test_top_shows_a_why_column(isolated_dirs: dict[str, Path]) -> None:
    _insert(
        [
            _row(
                platform="hackerone",
                handle="why",
                name="Why",
                bounty_max=5000,
                score=67.5,
                assets=["URL"],
            )
        ]
    )
    with connect() as conn:
        conn.execute(
            "UPDATE programs SET score_components = ?, extras = ? WHERE handle = ?",
            (
                json.dumps({"skill_fit": 0.67, "breadth": 0.81, "pays": 1.0}),
                json.dumps({"response_efficiency_p": 80}),
                "why",
            ),
        )
    result = runner.invoke(app, ["top"], env={"COLUMNS": "240"})
    assert result.exit_code == 0, result.output
    collapsed = "".join(result.output.split())
    assert "fit0.67" in collapsed
    assert "breadth0.81" in collapsed
    assert "H1responsep80" in collapsed
    assert "67.5" in collapsed


def test_top_uses_readonly_connection(isolated_dirs: dict[str, Path]) -> None:
    _insert(
        [
            _row(
                platform="hackerone",
                handle="x",
                name="X",
                bounty_max=100,
                score=1.0,
                assets=["URL"],
            )
        ]
    )
    result = runner.invoke(app, ["top"])
    assert result.exit_code == 0, result.output
    with sqlite3.connect(db_path()) as conn:
        count = conn.execute("SELECT COUNT(*) FROM programs").fetchone()[0]
    assert count == 1
