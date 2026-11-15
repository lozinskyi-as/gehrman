"""List the highest-scored programs from the last `gehrman update`.

Opens the SQLite store read-only and prints a Rich table. Every catalog
string is escaped before rendering — program names and asset types are
conditionally trusted (see SECURITY.md) and must not be interpreted as
Rich markup.
"""

from __future__ import annotations

import json
import sqlite3
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from gehrman.catalogs.store import db_path
from gehrman.constants import Platform

console = Console()


def top_command(
    limit: Annotated[
        int,
        typer.Option("--limit", "-n", min=1, max=100, help="Number of rows to show."),
    ] = 10,
    platform: Annotated[
        Platform | None,
        typer.Option("--platform", case_sensitive=False, help="Restrict to one platform."),
    ] = None,
    min_bounty: Annotated[
        float | None,
        typer.Option("--min-bounty", help="Only show programs with bounty_max >= this amount."),
    ] = None,
) -> None:
    """Show the highest-scored programs from the local snapshot."""
    path = db_path()
    if not path.is_file():
        console.print(f"[red]No database at {path}. Run `gehrman update` first.[/red]")
        raise typer.Exit(code=1)

    query = (
        "SELECT platform, handle, name, url, bounty_max, asset_types, "
        "in_scope_count, score FROM programs WHERE score > 0"
    )
    args: list[object] = []
    if platform is not None:
        query += " AND platform = ?"
        args.append(platform.value)
    if min_bounty is not None:
        query += " AND bounty_max IS NOT NULL AND bounty_max >= ?"
        args.append(min_bounty)
    # SQLite treats NULL as the smallest value, so DESC puts NULL bounties last.
    query += " ORDER BY score DESC, bounty_max DESC, handle ASC LIMIT ?"
    args.append(limit)

    uri = f"file:{path}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        rows = conn.execute(query, args).fetchall()
    finally:
        conn.close()

    if not rows:
        console.print(
            "[yellow]No matching programs. Run `gehrman update` or widen the filters.[/yellow]"
        )
        return

    table = Table(title=f"Top {len(rows)} programs by score")
    table.add_column("Score", justify="right")
    table.add_column("Platform")
    table.add_column("Program")
    table.add_column("Bounty $", justify="right")
    table.add_column("Scope", justify="right")
    table.add_column("Assets")
    table.add_column("URL")
    for plat, handle, name, url, bounty_max, assets_json, in_scope, score in rows:
        assets = ", ".join(json.loads(assets_json)[:4])
        bounty_display = f"{bounty_max:,.0f}" if bounty_max else "—"
        display_name = name or handle
        table.add_row(
            f"{score:.2f}",
            escape(plat),
            escape(display_name),
            bounty_display,
            str(in_scope),
            escape(assets),
            escape(url),
        )
    console.print(table)
