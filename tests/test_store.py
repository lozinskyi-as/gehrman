from __future__ import annotations

import sqlite3
from pathlib import Path

from gehrman.catalogs.store import SCHEMA_VERSION, connect, db_path


def test_stale_database_is_rebuilt(isolated_dirs: dict[str, Path]) -> None:
    path = db_path()
    path.parent.mkdir(parents=True)
    with sqlite3.connect(path) as conn:
        conn.execute(
            """
            CREATE TABLE programs (
              platform TEXT NOT NULL,
              handle TEXT NOT NULL,
              name TEXT NOT NULL,
              PRIMARY KEY (platform, handle)
            )
            """
        )
        conn.execute(
            "INSERT INTO programs (platform, handle, name) VALUES ('hackerone', 'old', 'Old')"
        )
        conn.execute("PRAGMA user_version = 0")

    with connect() as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(programs)")}
        assert "in_scope_assets" in columns
        assert "out_of_scope_assets" in columns
        assert "response_efficiency_percentage" in columns
        assert conn.execute("SELECT COUNT(*) FROM programs").fetchone()[0] == 0
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        assert version == SCHEMA_VERSION
        conn.execute(
            """
            INSERT INTO programs (
              platform, handle, name, url, offers_bounty, asset_types,
              in_scope_count, out_of_scope_count, policy_excerpt, score,
              fetched_at, snapshot_sha256, snapshot_path
            ) VALUES ('hackerone', 'kept', 'Kept', 'https://example.test', 1, '[]',
                      0, 0, '', 1, 't', 'abc', '/tmp/s.json')
            """
        )

    with connect() as conn:
        assert conn.execute("SELECT handle FROM programs").fetchone()[0] == "kept"
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
