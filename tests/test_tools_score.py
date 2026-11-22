"""Agent tools must expose score components and extras as seeded numbers."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from gehrman.agent.defenses import collect_allowed_from_result, number_check
from gehrman.agent.tools import dispatch
from gehrman.catalogs.models import Program, ScopeAsset
from gehrman.catalogs.store import connect, replace_platform_programs
from gehrman.config import AppConfig, ProfileConfig
from gehrman.constants import AssetType, Platform, Skill
from gehrman.scoring import ScoreResult


def _config() -> AppConfig:
    return AppConfig(profile=ProfileConfig(skills=[Skill.WEB], bounty_only=True))


def _seed() -> None:
    program = Program(
        platform=Platform.HACKERONE,
        handle="acme",
        name="Acme",
        url="https://hackerone.com/acme",
        offers_bounty=True,
        bounty_max=5000,
        in_scope_assets=[ScopeAsset(identifier="acme.example", type=AssetType.WEB)],
        in_scope_count=1,
    )
    result = ScoreResult(
        score=67.5,
        components={"skill_fit": 0.67, "breadth": 0.81, "pays": 1.0},
        extras={"response_efficiency_p": 80, "bounty_eligible_share": 0.42},
    )
    with connect() as conn:
        replace_platform_programs(
            conn,
            program.platform,
            [(program, result)],
            datetime.now(UTC),
            Path("/tmp/snap.json"),
            "abc",
        )


def test_get_and_search_seed_component_numbers(isolated_dirs: dict[str, Path]) -> None:
    del isolated_dirs
    _seed()
    config = _config()
    got, _ = dispatch("get", {"platform": "hackerone", "handle": "acme"}, config)
    found, _ = dispatch("search", {"platform": "hackerone", "limit": 5}, config)
    assert got["score_components"]["skill_fit"] == 0.67
    assert got["extras"]["response_efficiency_p"] == 80
    assert got["why"].startswith("fit 0.67")
    assert found["results"][0]["score_components"]["breadth"] == 0.81
    assert "policy_excerpt" not in got

    hosts: set[str] = set()
    numbers: set[str] = set()
    collect_allowed_from_result(got, hosts, numbers)
    answer = "fit 0.67, breadth 0.81, score 67.5, response 80, eligible 0.42, bounty 5000"
    ok, evidence = number_check(answer, enabled=True, allowed_numbers=numbers)
    assert ok, evidence
    blocked, _ = number_check("The hidden figure is 99999.", enabled=True, allowed_numbers=numbers)
    assert blocked is False
