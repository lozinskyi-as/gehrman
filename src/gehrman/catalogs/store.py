"""Persist raw snapshots and normalized programs.

Layout under `data_dir()`:
  gehrman.db              SQLite, writable by this process only
  snapshots/<platform>/<timestamp>_<sha12>.json  raw bodies (0600)

Each `update` run replaces the rows for the platforms it fetched. Prior
rows for untouched platforms stay intact. Raw snapshots are kept on disk
so the thesis experiment can replay the exact bytes that produced a
score, which is why we persist the SHA-256 of the body with every row.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from gehrman.catalogs.models import Program
from gehrman.constants import Platform
from gehrman.paths import data_dir, ensure_private_dir

SCHEMA = """
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
  PRIMARY KEY (platform, handle)
);
CREATE INDEX IF NOT EXISTS idx_programs_score ON programs(score DESC);
"""


def db_path() -> Path:
    return data_dir() / "gehrman.db"


def snapshot_dir() -> Path:
    return data_dir() / "snapshots"


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    ensure_private_dir(data_dir())
    conn = sqlite3.connect(db_path())
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def write_snapshot(platform: Platform, body: bytes, fetched_at: datetime) -> tuple[Path, str]:
    directory = snapshot_dir() / platform.value
    ensure_private_dir(directory)
    sha = hashlib.sha256(body).hexdigest()
    timestamp = fetched_at.strftime("%Y%m%dT%H%M%SZ")
    path = directory / f"{timestamp}_{sha[:12]}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_bytes(body)
    tmp.chmod(0o600)
    tmp.replace(path)
    return path, sha


def replace_platform_programs(
    conn: sqlite3.Connection,
    platform: Platform,
    scored: list[tuple[Program, float]],
    fetched_at: datetime,
    snapshot_path: Path,
    snapshot_sha256: str,
) -> None:
    conn.execute("DELETE FROM programs WHERE platform = ?", (platform.value,))
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
        [
            (
                program.platform.value,
                program.handle,
                program.name,
                program.url,
                int(program.offers_bounty),
                program.bounty_min,
                program.bounty_max,
                json.dumps(program.asset_types),
                program.in_scope_count,
                program.out_of_scope_count,
                program.submission_state,
                program.average_time_to_first_response,
                program.average_time_to_bounty_awarded,
                program.average_time_to_report_resolved,
                program.policy_excerpt,
                score,
                fetched_at.isoformat(),
                snapshot_sha256,
                str(snapshot_path),
            )
            for program, score in scored
        ],
    )
