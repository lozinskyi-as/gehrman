from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from gehrman.catalogs.store import db_path, snapshot_dir
from gehrman.cli import app
from gehrman.config import AppConfig, ProfileConfig, save_config
from gehrman.constants import CATALOG_URLS, Platform, Skill

runner = CliRunner()


def _write_config() -> None:
    save_config(
        AppConfig(
            profile=ProfileConfig(skills=[Skill.WEB], bounty_only=True),
        )
    )


def _fake_bodies() -> dict[str, bytes]:
    h1 = [
        {
            "handle": "acme",
            "name": "Acme",
            "url": "https://hackerone.com/acme",
            "submission_state": "open",
            "offers_bounties": True,
            "policy": "Be nice.",
            "average_time_to_first_program_response": 6.0,
            "response_efficiency_percentage": 90,
            "targets": {
                "in_scope": [
                    {
                        "asset_type": "URL",
                        "asset_identifier": "acme.com",
                        "eligible_for_bounty": True,
                        "instruction": "do not let this text affect storage",
                    }
                ],
                "out_of_scope": [],
            },
        },
        {
            # Invalid: missing name — should be skipped, not crash.
            "handle": "broken",
            "url": "https://hackerone.com/broken",
        },
    ]
    bugcrowd = [
        {
            "name": "Beta",
            "url": "https://bugcrowd.com/beta",
            "max_payout": 5000.0,
            "targets": {
                "in_scope": [{"type": "website", "target": "beta.example", "uri": "beta.example"}],
                "out_of_scope": [],
            },
        }
    ]
    intigriti = [
        {
            "id": "gamma",
            "name": "Gamma",
            "url": "https://app.intigriti.com/programs/x/gamma/detail",
            "min_bounty": {"value": 250.0, "currency": "EUR"},
            "max_bounty": {"value": 2500.0, "currency": "EUR"},
            "targets": {
                "in_scope": [
                    {
                        "type": "url",
                        "endpoint": "api.gamma",
                        "description": "free text must not be stored on the asset",
                    }
                ],
                "out_of_scope": [],
            },
        }
    ]
    return {
        CATALOG_URLS[Platform.HACKERONE]: json.dumps(h1).encode(),
        CATALOG_URLS[Platform.BUGCROWD]: json.dumps(bugcrowd).encode(),
        CATALOG_URLS[Platform.INTIGRITI]: json.dumps(intigriti).encode(),
    }


@pytest.fixture
def fake_fetch(monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    bodies = _fake_bodies()

    def fake(url: str, **_: Any) -> bytes:
        try:
            return bodies[url]
        except KeyError as exc:
            raise AssertionError(f"unexpected URL: {url}") from exc

    monkeypatch.setattr("gehrman.commands.update.fetch_catalog", fake)
    return bodies


def test_update_writes_snapshots_and_rows(
    isolated_dirs: dict[str, Path], fake_fetch: dict[str, bytes]
) -> None:
    _write_config()
    result = runner.invoke(app, ["update"])
    assert result.exit_code == 0, result.output

    assert db_path().is_file()
    with sqlite3.connect(db_path()) as conn:
        rows = conn.execute(
            "SELECT platform, handle, offers_bounty, score FROM programs ORDER BY platform, handle"
        ).fetchall()
        assert [r[0] for r in rows] == ["bugcrowd", "hackerone", "intigriti"]
        assert all(r[2] == 1 for r in rows)
        assert all(r[3] > 0 for r in rows)

        acme = conn.execute(
            "SELECT average_time_to_first_response, response_efficiency_percentage, "
            "in_scope_assets FROM programs WHERE platform = 'hackerone'"
        ).fetchone()
        assert acme[0] == 6.0
        assert acme[1] == 90
        acme_assets = json.loads(acme[2])
        assert acme_assets == [{"identifier": "acme.com", "type": "web", "bounty_eligible": True}]
        assert "instruction" not in acme[2]
        components = conn.execute(
            "SELECT score_components, extras FROM programs WHERE platform = 'hackerone'"
        ).fetchone()
        assert json.loads(components[0])["skill_fit"] == 1.0
        assert "policy" not in components[0]
        h1_extras = json.loads(components[1])
        assert h1_extras["response_efficiency_p"] == 100
        assert h1_extras["first_response_p"] == 100
        assert h1_extras["bounty_eligible_share"] == 1
        assert h1_extras["new_assets_30d"] == 0
        assert h1_extras["new_program"] is False
        assert json.loads(components[0])["freshness"] == 0.0
        assert "6.0" not in components[1]
        assert "90" not in components[1]
        history = conn.execute(
            "SELECT identifier, baseline FROM asset_history "
            "WHERE platform = 'hackerone' AND in_scope = 1"
        ).fetchone()
        assert history == ("acme.com", 1)

        beta = conn.execute("SELECT extras FROM programs WHERE platform = 'bugcrowd'").fetchone()
        assert json.loads(beta[0])["max_payout"] == 5000

        gamma = conn.execute(
            "SELECT in_scope_assets FROM programs WHERE platform = 'intigriti'"
        ).fetchone()
        assert json.loads(gamma[0]) == [
            {"identifier": "api.gamma", "type": "web", "bounty_eligible": None}
        ]
        assert "description" not in gamma[0]

    for plat in Platform:
        files = list((snapshot_dir() / plat.value).glob("*.json"))
        assert len(files) == 1
        assert files[0].stat().st_mode & 0o777 == 0o600


def test_dry_run_does_not_change_asset_history(
    isolated_dirs: dict[str, Path], fake_fetch: dict[str, bytes]
) -> None:
    _write_config()
    assert runner.invoke(app, ["update"]).exit_code == 0
    with sqlite3.connect(db_path()) as conn:
        before = conn.execute(
            "SELECT platform, handle, identifier, in_scope, first_seen_at, last_seen_at, baseline "
            "FROM asset_history ORDER BY platform, handle, identifier, in_scope"
        ).fetchall()
    assert before
    result = runner.invoke(app, ["update", "--dry-run"])
    assert result.exit_code == 0, result.output
    with sqlite3.connect(db_path()) as conn:
        after = conn.execute(
            "SELECT platform, handle, identifier, in_scope, first_seen_at, last_seen_at, baseline "
            "FROM asset_history ORDER BY platform, handle, identifier, in_scope"
        ).fetchall()
    assert after == before


def test_update_dry_run_writes_nothing(
    isolated_dirs: dict[str, Path], fake_fetch: dict[str, bytes]
) -> None:
    _write_config()
    result = runner.invoke(app, ["update", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert not db_path().exists()
    assert not snapshot_dir().exists()


def test_update_single_platform(
    isolated_dirs: dict[str, Path], fake_fetch: dict[str, bytes]
) -> None:
    _write_config()
    result = runner.invoke(app, ["update", "--platform", "hackerone"])
    assert result.exit_code == 0, result.output
    with sqlite3.connect(db_path()) as conn:
        platforms = [r[0] for r in conn.execute("SELECT DISTINCT platform FROM programs")]
    assert platforms == ["hackerone"]


def test_update_without_config_fails_cleanly(isolated_dirs: dict[str, Path]) -> None:
    result = runner.invoke(app, ["update"])
    assert result.exit_code == 1
    # Rich wraps the path, so the two words may land on different lines.
    assert "gehrman init" in result.output.replace("\n", "")
