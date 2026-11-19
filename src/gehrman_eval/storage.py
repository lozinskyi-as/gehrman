"""Run directory layout, append-only results log, baselines cache.

The runner writes everything under `RUN_DIR`:

    RUN_DIR/
      manifest.json         # parameters, git commit, corpus/DB hashes
      base.db               # frozen copy of the catalog DB
      results.jsonl         # one TrialResult per line, append-only
      baselines.json        # cached clean-run top-3 for ranking goals
      trials/<trial_id>/    # per-trial DB copy (poisoned)

Append-only `results.jsonl` is the source of truth — the runner can be
killed mid-run and `--resume` picks up from whatever is on disk.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class Manifest:
    started_at: str
    gehrman_git_commit: str | None
    corpus_path: str
    corpus_sha256: str
    base_db_path: str
    base_db_sha256: str
    models: list[str]
    profiles: list[str]
    repeats: int
    seed: int
    temperature: float
    max_trials: int | None
    attack_count: int
    extra: dict[str, Any] = field(default_factory=dict)

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def git_commit(cwd: Path | None = None) -> str | None:
    try:
        output = subprocess.check_output(  # noqa: S603
            ["git", "rev-parse", "HEAD"],  # noqa: S607
            cwd=str(cwd) if cwd else None,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return output.decode().strip()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def timestamp_dirname() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def prepare_run_dir(parent: Path, *, subdir: str | None = None) -> Path:
    name = subdir or timestamp_dirname()
    run_dir = parent / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "trials").mkdir(exist_ok=True)
    return run_dir


def copy_base_db(source: Path, run_dir: Path) -> Path:
    if not source.is_file():
        raise FileNotFoundError(f"Base DB not found: {source}")
    base_dir = run_dir / "base"
    base_dir.mkdir(exist_ok=True)
    target = base_dir / "gehrman.db"
    shutil.copy2(source, target)
    return target


def materialize_trial_db(base_db: Path, run_dir: Path, trial_id: str) -> Path:
    trial_dir = run_dir / "trials" / trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    target = trial_dir / "gehrman.db"
    shutil.copy2(base_db, target)
    return target


def _append_jsonl(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip("\n") + "\n")
        handle.flush()


def append_result(run_dir: Path, line: str) -> None:
    """Append a JSON line to results.jsonl with fsync for crash safety."""
    _append_jsonl(run_dir / "results.jsonl", line)


def append_control_result(run_dir: Path, line: str) -> None:
    _append_jsonl(run_dir / "controls.jsonl", line)


def _completed_ids_from(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = record.get("trial_id")
            if isinstance(tid, str):
                done.add(tid)
    return done


def completed_trial_ids(run_dir: Path) -> set[str]:
    return _completed_ids_from(run_dir / "results.jsonl")


def completed_control_ids(run_dir: Path) -> set[str]:
    return _completed_ids_from(run_dir / "controls.jsonl")


def read_baselines(run_dir: Path) -> dict[str, list[str]]:
    path = run_dir / "baselines.json"
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def write_baselines(run_dir: Path, baselines: dict[str, list[str]]) -> None:
    path = run_dir / "baselines.json"
    path.write_text(json.dumps(baselines, indent=2), encoding="utf-8")


def baseline_key(*, user_query: str, model: str, profile: str) -> str:
    return hashlib.sha256(
        f"{user_query}|{model}|{profile}".encode()
    ).hexdigest()[:16]


def known_handles(db_path: Path) -> list[str]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT handle FROM programs").fetchall()
    finally:
        conn.close()
    return [row[0] for row in rows]
