from __future__ import annotations

from gehrman.catalogs.models import Program
from gehrman.config import ProfileConfig
from gehrman.constants import Platform, Skill
from gehrman.scoring import score_program


def _program(**overrides: object) -> Program:
    base: dict[str, object] = {
        "platform": Platform.HACKERONE,
        "handle": "acme",
        "name": "Acme",
        "url": "https://hackerone.com/acme",
        "offers_bounty": True,
        "bounty_max": 1000.0,
        "asset_types": ["URL"],
        "in_scope_count": 3,
        "out_of_scope_count": 1,
    }
    base.update(overrides)
    return Program(**base)  # type: ignore[arg-type]


def _profile(**overrides: object) -> ProfileConfig:
    base: dict[str, object] = {"skills": [Skill.WEB], "bounty_only": True}
    base.update(overrides)
    return ProfileConfig(**base)  # type: ignore[arg-type]


def test_bounty_only_filters_non_bounty_programs() -> None:
    program = _program(offers_bounty=False, bounty_max=None)
    assert score_program(program, _profile()) == 0.0


def test_matching_skill_increases_score() -> None:
    matching = score_program(_program(asset_types=["URL"]), _profile(skills=[Skill.WEB]))
    missing = score_program(_program(asset_types=["CIDR"]), _profile(skills=[Skill.WEB]))
    assert matching > missing


def test_score_is_deterministic() -> None:
    program = _program()
    profile = _profile()
    assert score_program(program, profile) == score_program(program, profile)


def test_higher_bounty_scores_higher() -> None:
    low = score_program(_program(bounty_max=100.0), _profile())
    high = score_program(_program(bounty_max=100000.0), _profile())
    assert high > low
