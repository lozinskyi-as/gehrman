"""Shared fixtures. Tests never touch the real home directory."""

from __future__ import annotations

from pathlib import Path

import pytest

from gehrman.constants import CONFIG_DIR_ENV, DATA_DIR_ENV


@pytest.fixture
def isolated_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """Point config and data paths at a temporary tree."""
    config = tmp_path / "config"
    data = tmp_path / "data"
    monkeypatch.setenv(CONFIG_DIR_ENV, str(config))
    monkeypatch.setenv(DATA_DIR_ENV, str(data))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.delenv("XDG_DATA_HOME", raising=False)
    return {"config": config, "data": data}
