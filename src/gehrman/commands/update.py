"""Download, validate, store, and score public program catalogs.

The command is passive: it only talks to the static CDN that hosts the
three catalog JSON files. It never contacts a program's targets.

Untrusted free-text (policy excerpts) is persisted but never rendered in
the summary table — the table only shows counts.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from gehrman.catalogs.fetch import CatalogFetchError, fetch_catalog
from gehrman.catalogs.models import (
    BugcrowdProgram,
    HackerOneProgram,
    IntigritiProgram,
    Program,
)
from gehrman.catalogs.normalize import (
    normalize_bugcrowd,
    normalize_hackerone,
    normalize_intigriti,
)
from gehrman.catalogs.store import connect, replace_platform_programs, write_snapshot
from gehrman.config import load_config
from gehrman.constants import CATALOG_URLS, Platform
from gehrman.scoring import score_program

console = Console()

NORMALIZERS = {
    Platform.HACKERONE: (HackerOneProgram, normalize_hackerone),
    Platform.BUGCROWD: (BugcrowdProgram, normalize_bugcrowd),
    Platform.INTIGRITI: (IntigritiProgram, normalize_intigriti),
}


def _process_platform(
    plat: Platform, body: bytes, truncate: int, profile
) -> tuple[list[tuple[Program, float]], int]:
    try:
        raw = json.loads(body)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"{plat.value}: invalid JSON ({exc})") from exc
    if not isinstance(raw, list):
        raise typer.BadParameter(f"{plat.value}: expected a JSON array")

    model_cls, normalizer = NORMALIZERS[plat]
    scored: list[tuple[Program, float]] = []
    skipped = 0
    for entry in raw:
        try:
            validated = model_cls.model_validate(entry)
        except ValidationError:
            skipped += 1
            continue
        program = normalizer(validated, truncate)
        scored.append((program, score_program(program, profile)))
    return scored, skipped


def update_command(
    platform: Annotated[
        list[Platform] | None,
        typer.Option(
            "--platform",
            case_sensitive=False,
            help="Repeatable. Default: all three platforms.",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Fetch and parse, but do not write the database or snapshots.",
        ),
    ] = False,
) -> None:
    """Download public program catalogs, store snapshots, and compute scores."""
    try:
        config = load_config()
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    selected = list(dict.fromkeys(platform)) if platform else list(Platform)
    truncate = config.agent.text_truncate
    fetched_at = datetime.now(UTC)

    rows: list[tuple[str, int, int, int]] = []
    for plat in selected:
        url = CATALOG_URLS[plat]
        console.print(f"Fetching {plat.value} …")
        try:
            body = fetch_catalog(url)
        except CatalogFetchError as exc:
            console.print(f"[red]{plat.value}: {exc}[/red]")
            raise typer.Exit(code=1) from exc

        scored, skipped = _process_platform(plat, body, truncate, config.profile)
        if skipped:
            console.print(
                f"[yellow]{plat.value}: skipped {skipped} entries that failed validation[/yellow]"
            )

        if not dry_run:
            snapshot_path, sha = write_snapshot(plat, body, fetched_at)
            with connect() as conn:
                replace_platform_programs(conn, plat, scored, fetched_at, snapshot_path, sha)

        nonzero = sum(1 for _, s in scored if s > 0)
        rows.append((plat.value, len(scored), skipped, nonzero))

    table = Table(title="Gehrman update summary")
    table.add_column("Platform")
    table.add_column("Programs", justify="right")
    table.add_column("Skipped", justify="right")
    table.add_column("Scored > 0", justify="right")
    for name, count, skipped, nonzero in rows:
        table.add_row(name, str(count), str(skipped), str(nonzero))
    console.print(table)
    if dry_run:
        console.print("[yellow]--dry-run: nothing was written to disk.[/yellow]")
