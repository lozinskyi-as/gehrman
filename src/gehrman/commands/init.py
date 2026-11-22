"""Interactive (and scriptable) first-run setup.

Collects the researcher profile and the local Ollama model to use.
Gehrman is local-only, so no API keys are prompted for or stored.
"""

from __future__ import annotations

from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel

from gehrman.config import AgentConfig, AppConfig, ProfileConfig, save_config
from gehrman.constants import (
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    SKILL_VALUES,
    Skill,
)
from gehrman.ollama import OllamaUnavailableError, list_models
from gehrman.paths import config_path

console = Console()


def _prompt_skills() -> list[Skill]:
    raw = typer.prompt(f"Skills, comma-separated. Available: {', '.join(SKILL_VALUES)}")
    parts = [part.strip().lower() for part in raw.split(",") if part.strip()]
    if not parts:
        raise typer.BadParameter("Select at least one skill")
    try:
        return [Skill(part) for part in parts]
    except ValueError as exc:
        raise typer.BadParameter(f"Skills must be from: {', '.join(SKILL_VALUES)}") from exc


def _prompt_model(base_url: str, default: str) -> str:
    try:
        models = list_models(base_url)
    except OllamaUnavailableError as exc:
        console.print(f"[yellow]Could not reach Ollama at {base_url}: {exc}[/yellow]")
        return typer.prompt("Enter a model name manually", default=default)

    if not models:
        console.print(f"[yellow]Ollama at {base_url} has no models pulled.[/yellow]")
        return typer.prompt("Enter a model name manually", default=default)

    console.print("Available local Ollama models:")
    for index, name in enumerate(models, start=1):
        console.print(f"  {index}. {name}")
    while True:
        raw = typer.prompt("Choose a model (number)", default="1")
        try:
            choice = int(raw)
        except ValueError:
            console.print(f"[red]Enter a number between 1 and {len(models)}.[/red]")
            continue
        if 1 <= choice <= len(models):
            return models[choice - 1]
        console.print(f"[red]Enter a number between 1 and {len(models)}.[/red]")


def init_command(
    skill: Annotated[
        list[Skill] | None,
        typer.Option(
            "--skill", case_sensitive=False, help="Repeatable. From the fixed skill list."
        ),
    ] = None,
    bounty_only: Annotated[
        bool | None,
        typer.Option(
            "--bounty-only/--all-programs",
            help="Only recommend programs that offer bounties.",
        ),
    ] = None,
    base_url: Annotated[
        str | None,
        typer.Option(
            "--base-url",
            help=f"Ollama OpenAI-compatible endpoint. Default: {DEFAULT_OLLAMA_BASE_URL}.",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "--model",
            help="Ollama model identifier. Skips the interactive menu.",
        ),
    ] = None,
) -> None:
    """Create ~/.config/gehrman/config.toml."""
    target = config_path()
    if target.is_file():
        if not typer.confirm(f"Config already exists at {target}. Overwrite?", default=False):
            console.print("Keeping the existing configuration.")
            raise typer.Exit(code=1)

    resolved_skills = skill or _prompt_skills()
    if bounty_only is None:
        resolved_bounty_only = typer.confirm(
            "Only recommend programs that offer bounties?",
            default=True,
        )
    else:
        resolved_bounty_only = bounty_only

    resolved_base_url = base_url or DEFAULT_OLLAMA_BASE_URL
    interactive = skill is None
    if model:
        resolved_model = model
    elif interactive:
        resolved_model = _prompt_model(resolved_base_url, DEFAULT_OLLAMA_MODEL)
    else:
        resolved_model = DEFAULT_OLLAMA_MODEL

    try:
        profile = ProfileConfig(
            skills=resolved_skills,
            bounty_only=resolved_bounty_only,
        )
        agent = AgentConfig(base_url=resolved_base_url, model=resolved_model)
        config = AppConfig(profile=profile, agent=agent)
    except ValidationError as exc:
        raise typer.BadParameter(str(exc)) from exc

    written = save_config(config)
    console.print(f"Wrote configuration to {written}")

    skills_display = ", ".join(item.value for item in config.profile.skills)
    summary = (
        f"Skills     : {skills_display}\n"
        f"Bounty only: {config.profile.bounty_only}\n"
        f"Ollama     : {config.agent.base_url} ({config.agent.model})\n"
        f"Config     : {written}"
    )
    console.print(Panel(summary, title="Gehrman is ready", border_style="green"))
