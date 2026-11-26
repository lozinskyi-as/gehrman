"""Fixtures shared by the eval-harness unit tests."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pytest


def _write_program_row(
    conn: sqlite3.Connection,
    *,
    platform: str,
    handle: str,
    policy: str = "",
    score: float = 5.0,
    url: str | None = None,
    bounty_max: float | None = 10000.0,
) -> None:
    conn.execute(
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
        (
            platform,
            handle,
            handle.capitalize(),
            url or f"https://{handle}.example.com",
            1 if bounty_max else 0,
            None,
            bounty_max,
            json.dumps(["URL"]),
            1,
            0,
            None,
            None,
            None,
            None,
            policy,
            score,
            datetime.now(UTC).isoformat(),
            "deadbeef",
            "/tmp/snap.json",
        ),
    )


@pytest.fixture
def seed_db(tmp_path: Path):
    """Return a factory that builds a fresh SQLite DB with the given programs."""
    db_path = tmp_path / "eval.db"

    def _seed(programs: list[dict]) -> Path:
        with sqlite3.connect(db_path) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS programs (
                  platform TEXT NOT NULL,
                  handle TEXT NOT NULL,
                  name TEXT NOT NULL,
                  url TEXT NOT NULL,
                  offers_bounty INTEGER NOT NULL,
                  bounty_min REAL,
                  bounty_max REAL,
                  asset_types TEXT NOT NULL,
                  in_scope_count INTEGER NOT NULL,
                  out_of_scope_count INTEGER NOT NULL,
                  submission_state TEXT,
                  average_time_to_first_response REAL,
                  average_time_to_bounty_awarded REAL,
                  average_time_to_report_resolved REAL,
                  policy_excerpt TEXT NOT NULL,
                  score REAL NOT NULL,
                  fetched_at TEXT NOT NULL,
                  snapshot_sha256 TEXT NOT NULL,
                  snapshot_path TEXT NOT NULL,
                  score_components TEXT NOT NULL DEFAULT '{}',
                  extras TEXT NOT NULL DEFAULT '{}',
                  PRIMARY KEY (platform, handle)
                );
                """
            )
            conn.execute("DELETE FROM programs")
            for program in programs:
                _write_program_row(conn, **program)
            conn.commit()
        return db_path

    return _seed
