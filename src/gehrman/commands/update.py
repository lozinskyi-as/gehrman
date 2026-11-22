"""Download, validate, store, and score public program catalogs.

The command is passive: it only talks to the static CDN that hosts the
three catalog JSON files. It never contacts a program's targets.

Untrusted free-text (policy excerpts) is persisted but never rendered in
the summary table — the table only shows counts.
"""

from __future__ import annotations

import json
import sqlite3
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
from gehrman.catalogs.store import (
    SCHEMA_VERSION,
    connect,
    db_path,
    replace_platform_programs,
    write_snapshot,
)
from gehrman.config import load_config
from gehrman.constants import CATALOG_URLS, Platform
from gehrman.scoring import ScoreResult, score_programs

console = Console()

NORMALIZERS = {
    Platform.HACKERONE: (HackerOneProgram, normalize_hackerone),
    Platform.BUGCROWD: (BugcrowdProgram, normalize_bugcrowd),
    Platform.INTIGRITI: (IntigritiProgram, normalize_intigriti),
}


def _parse_platform(plat: Platform, body: bytes, truncate: int) -> tuple[list[Program], int]:
    try:
        raw = json.loads(body)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"{plat.value}: invalid JSON ({exc})") from exc
    if not isinstance(raw, list):
        raise typer.BadParameter(f"{plat.value}: expected a JSON array")

    model_cls, normalizer = NORMALIZERS[plat]
    programs: list[Program] = []
    skipped = 0
    for entry in raw:
        try:
            validated = model_cls.model_validate(entry)
        except ValidationError:
            skipped += 1
            continue
        programs.append(normalizer(validated, truncate))
    return programs, skipped


def _schema_is_stale() -> bool:
    path = db_path()
    if not path.is_file():
        return False
    conn = sqlite3.connect(path)
    try:
        row = conn.execute("PRAGMA user_version").fetchone()
    finally:
        conn.close()
    current = int(row[0]) if row else 0
    return current != SCHEMA_VERSION


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
    if not dry_run and _schema_is_stale():
        console.print(
            "[yellow]Schema changed; the local database will be rebuilt by this update.[/yellow]"
        )
        if set(selected) != set(Platform):
            console.print(
                "[yellow]Only the selected platforms are refilled. "
                "Run `gehrman update` with no --platform to restore the others.[/yellow]"
            )

    fetched: list[tuple[Platform, bytes, list[Program], int]] = []
    for plat in selected:
        url = CATALOG_URLS[plat]
        console.print(f"Fetching {plat.value} …")
        try:
            body = fetch_catalog(url)
        except CatalogFetchError as exc:
            console.print(f"[red]{plat.value}: {exc}[/red]")
            raise typer.Exit(code=1) from exc

        programs, skipped = _parse_platform(plat, body, truncate)
        if skipped:
            console.print(
                f"[yellow]{plat.value}: skipped {skipped} entries that failed validation[/yellow]"
            )
        fetched.append((plat, body, programs, skipped))

    # Breadth is a percentile across every program fetched in this run.
    # A partial --platform update therefore ranks inside that subset.
    all_programs = [program for _, _, programs, _ in fetched for program in programs]
    scored_all = score_programs(all_programs, config.profile)
    by_key: dict[tuple[Platform, str], ScoreResult] = {
        (program.platform, program.handle): result
        for program, result in zip(all_programs, scored_all, strict=True)
    }

    rows: list[tuple[str, int, int, int]] = []
    for plat, body, programs, skipped in fetched:
        scored = [
            (
                program,
                by_key[(program.platform, program.handle)].score,
                by_key[(program.platform, program.handle)].components,
            )
            for program in programs
        ]
        if not dry_run:
            snapshot_path, sha = write_snapshot(plat, body, fetched_at)
            with connect() as conn:
                replace_platform_programs(conn, plat, scored, fetched_at, snapshot_path, sha)

        nonzero = sum(1 for _, score, _ in scored if score > 0)
        rows.append((plat.value, len(programs), skipped, nonzero))

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
