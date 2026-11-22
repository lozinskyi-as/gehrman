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
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from gehrman.catalogs.models import Program, ScopeAsset
from gehrman.constants import Platform
from gehrman.paths import data_dir, ensure_private_dir
from gehrman.scoring import ScoreResult

# Bump this when the tables change. `connect()` drops and recreates the
# database so the next `gehrman update` rebuilds it from snapshots.
SCHEMA_VERSION = 5

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
  in_scope_assets TEXT NOT NULL DEFAULT '[]',
  out_of_scope_assets TEXT NOT NULL DEFAULT '[]',
  in_scope_count INTEGER NOT NULL,
  out_of_scope_count INTEGER NOT NULL,
  submission_state TEXT,
  average_time_to_first_response REAL,
  average_time_to_bounty_awarded REAL,
  average_time_to_report_resolved REAL,
  response_efficiency_percentage REAL,
  policy_excerpt TEXT NOT NULL,
  score REAL NOT NULL,
  score_components TEXT NOT NULL DEFAULT '{}',
  extras TEXT NOT NULL DEFAULT '{}',
  fetched_at TEXT NOT NULL,
  snapshot_sha256 TEXT NOT NULL,
  snapshot_path TEXT NOT NULL,
  PRIMARY KEY (platform, handle)
);
CREATE INDEX IF NOT EXISTS idx_programs_score ON programs(score DESC);
CREATE TABLE IF NOT EXISTS asset_history (
  platform TEXT NOT NULL,
  handle TEXT NOT NULL,
  identifier TEXT NOT NULL,
  in_scope INTEGER NOT NULL,
  first_seen_at TEXT NOT NULL,
  last_seen_at TEXT NOT NULL,
  baseline INTEGER NOT NULL,
  PRIMARY KEY (platform, handle, identifier, in_scope)
);
"""


def db_path() -> Path:
    return data_dir() / "gehrman.db"


def snapshot_dir() -> Path:
    return data_dir() / "snapshots"


def schema_is_current(conn: sqlite3.Connection) -> bool:
    row = conn.execute("PRAGMA user_version").fetchone()
    current = int(row[0]) if row else 0
    return current == SCHEMA_VERSION


def _apply_schema(conn: sqlite3.Connection) -> bool:
    """Create tables, or rebuild them when `user_version` is stale.

    Returns True when an existing database was wiped because its schema
    version did not match. A fresh database also reports True so the
    version pragma is written, but there was nothing to keep.
    """
    rebuilt = not schema_is_current(conn)
    if rebuilt:
        conn.executescript("DROP TABLE IF EXISTS programs; DROP TABLE IF EXISTS asset_history;")
    conn.executescript(SCHEMA)
    if rebuilt:
        # PRAGMA does not accept bound parameters. SCHEMA_VERSION is a constant.
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    return rebuilt


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    ensure_private_dir(data_dir())
    conn = sqlite3.connect(db_path())
    try:
        _apply_schema(conn)
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


def _dump_assets(assets: list[ScopeAsset]) -> str:
    return json.dumps([asset.model_dump(mode="json") for asset in assets], separators=(",", ":"))


def replace_platform_programs(
    conn: sqlite3.Connection,
    platform: Platform,
    scored: list[tuple[Program, ScoreResult]],
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
          in_scope_assets, out_of_scope_assets,
          in_scope_count, out_of_scope_count, submission_state,
          average_time_to_first_response, average_time_to_bounty_awarded,
          average_time_to_report_resolved, response_efficiency_percentage,
          policy_excerpt, score, score_components, extras,
          fetched_at, snapshot_sha256, snapshot_path
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                _dump_assets(program.in_scope_assets),
                _dump_assets(program.out_of_scope_assets),
                program.in_scope_count,
                program.out_of_scope_count,
                program.submission_state,
                program.average_time_to_first_response,
                program.average_time_to_bounty_awarded,
                program.average_time_to_report_resolved,
                program.response_efficiency_percentage,
                program.policy_excerpt,
                result.score,
                json.dumps(result.components, separators=(",", ":")),
                json.dumps(result.extras, separators=(",", ":")),
                fetched_at.isoformat(),
                snapshot_sha256,
                str(snapshot_path),
            )
            for program, result in scored
        ],
    )


@dataclass(frozen=True)
class HistoryRow:
    """One scope asset seen in a stored catalog snapshot.

    `identifier=""` and `in_scope=0` is a program marker. It records that
    the handle existed at the baseline even when the program had no
    in-scope assets, so a later asset is new without the program itself
    looking new.
    """

    platform: str
    handle: str
    identifier: str
    in_scope: int
    first_seen_at: str
    last_seen_at: str
    baseline: int


def load_asset_history(conn: sqlite3.Connection) -> list[HistoryRow]:
    try:
        rows = conn.execute(
            """
            SELECT platform, handle, identifier, in_scope,
                   first_seen_at, last_seen_at, baseline
            FROM asset_history
            """
        ).fetchall()
    except sqlite3.OperationalError:
        return []
    return [
        HistoryRow(
            platform=row[0],
            handle=row[1],
            identifier=row[2],
            in_scope=int(row[3]),
            first_seen_at=row[4],
            last_seen_at=row[5],
            baseline=int(row[6]),
        )
        for row in rows
    ]


def apply_history(
    existing: list[HistoryRow],
    programs: list[Program],
    seen_at: datetime,
    *,
    window_days: int,
) -> tuple[list[HistoryRow], dict[tuple[str, str], tuple[int, bool]]]:
    """Fold `programs` into `existing` without writing.

    The first snapshot for a platform is the baseline: assets seen then
    get `baseline=1` and never count as new. Returns the merged rows and
    `(new_assets_30d, is_new_program)` for each program.
    """
    stamp = seen_at.isoformat()
    cutoff = (seen_at - timedelta(days=window_days)).isoformat()
    index: dict[tuple[str, str, str, int], HistoryRow] = {
        (row.platform, row.handle, row.identifier, row.in_scope): row for row in existing
    }
    platforms = {program.platform.value for program in programs}
    baseline_platforms = {
        platform for platform in platforms if not any(row.platform == platform for row in existing)
    }

    def touch(
        platform: str, handle: str, identifier: str, in_scope: int, *, is_baseline: bool
    ) -> None:
        key = (platform, handle, identifier, in_scope)
        current = index.get(key)
        if current is None:
            index[key] = HistoryRow(
                platform=platform,
                handle=handle,
                identifier=identifier,
                in_scope=in_scope,
                first_seen_at=stamp,
                last_seen_at=stamp,
                baseline=1 if is_baseline else 0,
            )
            return
        index[key] = HistoryRow(
            platform=current.platform,
            handle=current.handle,
            identifier=current.identifier,
            in_scope=current.in_scope,
            first_seen_at=current.first_seen_at,
            last_seen_at=stamp,
            baseline=current.baseline,
        )

    for program in programs:
        platform = program.platform.value
        is_baseline = platform in baseline_platforms
        touch(platform, program.handle, "", 0, is_baseline=is_baseline)
        seen: set[str] = set()
        for asset in program.in_scope_assets:
            identifier = asset.identifier.strip()
            if not identifier or identifier in seen:
                continue
            seen.add(identifier)
            touch(platform, program.handle, identifier, 1, is_baseline=is_baseline)

    stats: dict[tuple[str, str], tuple[int, bool]] = {}
    for program in programs:
        platform = program.platform.value
        handle = program.handle
        rows = [row for row in index.values() if row.platform == platform and row.handle == handle]
        new_assets = sum(
            1
            for row in rows
            if row.in_scope == 1 and row.baseline == 0 and row.first_seen_at >= cutoff
        )
        is_new = not any(row.baseline == 1 for row in rows)
        stats[(platform, handle)] = (new_assets, is_new)
    return list(index.values()), stats


def record_asset_history(
    conn: sqlite3.Connection,
    programs: list[Program],
    seen_at: datetime,
    *,
    window_days: int,
) -> dict[tuple[str, str], tuple[int, bool]]:
    """Upsert in-scope assets. `first_seen_at` and `baseline` are write-once."""
    existing = load_asset_history(conn)
    updated, stats = apply_history(existing, programs, seen_at, window_days=window_days)
    touched = {program.platform.value for program in programs}
    conn.executemany(
        """
        INSERT INTO asset_history (
          platform, handle, identifier, in_scope, first_seen_at, last_seen_at, baseline
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(platform, handle, identifier, in_scope) DO UPDATE SET
          last_seen_at = excluded.last_seen_at
        """,
        [
            (
                row.platform,
                row.handle,
                row.identifier,
                row.in_scope,
                row.first_seen_at,
                row.last_seen_at,
                row.baseline,
            )
            for row in updated
            if row.platform in touched
        ],
    )
    return stats


def preview_freshness(
    programs: list[Program],
    seen_at: datetime,
    *,
    window_days: int,
) -> dict[tuple[str, str], tuple[int, bool]]:
    """History delta for a dry run. Does not create or modify the database."""
    existing: list[HistoryRow] = []
    path = db_path()
    if path.is_file():
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            existing = load_asset_history(conn)
        finally:
            conn.close()
    _, stats = apply_history(existing, programs, seen_at, window_days=window_days)
    return stats
