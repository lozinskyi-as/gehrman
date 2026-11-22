from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from typer.testing import CliRunner

from gehrman.cli import app
from gehrman.config import FORBIDDEN_SECRET_KEYS, load_config
from gehrman.constants import DEFAULT_OLLAMA_BASE_URL, Skill
from gehrman.ollama import OllamaUnavailableError
from gehrman.paths import config_path

runner = CliRunner()


def _init_args() -> list[str]:
    return [
        "init",
        "--skill",
        "web",
        "--skill",
        "api",
        "--bounty-only",
    ]


def test_init_writes_config(isolated_dirs: dict[str, Path]) -> None:
    result = runner.invoke(app, _init_args())
    assert result.exit_code == 0, result.output
    config = load_config()
    assert config.profile.skills == [Skill.WEB, Skill.API]
    assert config.profile.bounty_only is True
    assert config.defenses.sanitize is True


def _collect_keys(payload: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(payload, dict):
        for key, value in payload.items():
            keys.add(str(key).lower())
            keys.update(_collect_keys(value))
    elif isinstance(payload, list):
        for item in payload:
            keys.update(_collect_keys(item))
    return keys


def test_init_config_has_no_secret_keys(isolated_dirs: dict[str, Path]) -> None:
    result = runner.invoke(app, _init_args())
    assert result.exit_code == 0, result.output
    with config_path().open("rb") as handle:
        data = tomllib.load(handle)
    assert _collect_keys(data).isdisjoint(FORBIDDEN_SECRET_KEYS)


def test_init_rejects_unknown_skill(isolated_dirs: dict[str, Path]) -> None:
    result = runner.invoke(app, ["init", "--skill", "mobile"])
    assert result.exit_code != 0


def test_init_with_model_flag(isolated_dirs: dict[str, Path]) -> None:
    result = runner.invoke(app, [*_init_args(), "--model", "digitsflow/bonsai-8b:latest"])
    assert result.exit_code == 0, result.output
    config = load_config()
    assert config.agent.model == "digitsflow/bonsai-8b:latest"


def test_init_with_base_url_flag(isolated_dirs: dict[str, Path]) -> None:
    result = runner.invoke(
        app,
        [*_init_args(), "--base-url", "http://ollama.local:11434/v1", "--model", "foo"],
    )
    assert result.exit_code == 0, result.output
    config = load_config()
    assert config.agent.base_url == "http://ollama.local:11434/v1"
    assert config.agent.model == "foo"


def test_init_interactive_menu_picks_discovered_model(
    isolated_dirs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "gehrman.commands.init.list_models",
        lambda base_url: ["llama3.1:latest", "digitsflow/bonsai-8b:latest"],
    )
    stdin = "\n".join(["web,api", "y", "2", ""]) + "\n"
    result = runner.invoke(app, ["init"], input=stdin)
    assert result.exit_code == 0, result.output
    assert "Available local Ollama models" in result.output
    config = load_config()
    assert config.agent.model == "digitsflow/bonsai-8b:latest"


def test_init_interactive_falls_back_when_ollama_unreachable(
    isolated_dirs: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(base_url: str) -> list[str]:
        raise OllamaUnavailableError("refused")

    monkeypatch.setattr("gehrman.commands.init.list_models", boom)
    stdin = "\n".join(["web", "y", "mymodel:latest"]) + "\n"
    result = runner.invoke(app, ["init"], input=stdin)
    assert result.exit_code == 0, result.output
    assert "Could not reach Ollama" in result.output
    assert load_config().agent.model == "mymodel:latest"


def test_init_default_base_url_when_no_flag(isolated_dirs: dict[str, Path]) -> None:
    result = runner.invoke(app, [*_init_args(), "--model", "x"])
    assert result.exit_code == 0, result.output
    assert load_config().agent.base_url == DEFAULT_OLLAMA_BASE_URL


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output
