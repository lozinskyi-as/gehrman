"""Append-only results log, baselines cache, manifest."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from gehrman_eval.storage import (
    append_result,
    baseline_key,
    completed_trial_ids,
    copy_base_db,
    file_sha256,
    known_handles,
    materialize_trial_db,
    prepare_run_dir,
    read_baselines,
    write_baselines,
)


def test_prepare_run_dir_creates_trials_subdir(tmp_path: Path) -> None:
    run_dir = prepare_run_dir(tmp_path, subdir="r1")
    assert run_dir.is_dir()
    assert (run_dir / "trials").is_dir()


def test_append_and_read_completed_trial_ids(tmp_path: Path) -> None:
    run_dir = prepare_run_dir(tmp_path, subdir="r1")
    append_result(run_dir, json.dumps({"trial_id": "aaaa"}))
    append_result(run_dir, json.dumps({"trial_id": "bbbb"}))
    append_result(run_dir, json.dumps({"trial_id": "aaaa"}))  # duplicate line
    done = completed_trial_ids(run_dir)
    assert done == {"aaaa", "bbbb"}


def test_completed_ids_empty_when_no_log(tmp_path: Path) -> None:
    run_dir = prepare_run_dir(tmp_path, subdir="r1")
    assert completed_trial_ids(run_dir) == set()


def test_completed_ids_skips_malformed_lines(tmp_path: Path) -> None:
    run_dir = prepare_run_dir(tmp_path, subdir="r1")
    (run_dir / "results.jsonl").write_text(
        'not json\n{"trial_id": "ok"}\n',
        encoding="utf-8",
    )
    assert completed_trial_ids(run_dir) == {"ok"}


def test_baselines_roundtrip(tmp_path: Path) -> None:
    run_dir = prepare_run_dir(tmp_path, subdir="r1")
    write_baselines(run_dir, {"abc": ["shopify", "github"]})
    assert read_baselines(run_dir) == {"abc": ["shopify", "github"]}


def test_baseline_key_is_stable() -> None:
    k1 = baseline_key(user_query="q", model="m", profile="p")
    k2 = baseline_key(user_query="q", model="m", profile="p")
    k3 = baseline_key(user_query="q", model="m", profile="other")
    assert k1 == k2
    assert k1 != k3


def test_copy_base_db_lands_as_gehrman_db(tmp_path: Path, seed_db) -> None:
    source = seed_db([{"platform": "hackerone", "handle": "x", "score": 1.0}])
    run_dir = prepare_run_dir(tmp_path, subdir="r1")
    target = copy_base_db(source, run_dir)
    assert target.name == "gehrman.db"
    assert target.parent.name == "base"
    assert target.is_file()


def test_materialize_trial_db_copies_file(tmp_path: Path, seed_db) -> None:
    source = seed_db([{"platform": "hackerone", "handle": "x", "score": 1.0}])
    run_dir = prepare_run_dir(tmp_path, subdir="r1")
    base = copy_base_db(source, run_dir)
    trial_db = materialize_trial_db(base, run_dir, trial_id="abcd1234")
    assert trial_db.is_file()
    with sqlite3.connect(trial_db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM programs").fetchone()[0] == 1


def test_known_handles(tmp_path: Path, seed_db) -> None:
    db = seed_db(
        [
            {"platform": "hackerone", "handle": "a", "score": 1.0},
            {"platform": "intigriti", "handle": "b", "score": 2.0},
        ]
    )
    assert set(known_handles(db)) == {"a", "b"}


def test_file_sha256_changes_with_content(tmp_path: Path) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("hello")
    b.write_text("world")
    assert file_sha256(a) != file_sha256(b)
