"""Interactive chat with the Gehrman agent over a local Ollama model.

The command is a thin shell around `agent.loop.run_turn`: it loads
config, confirms the SQLite snapshot exists, builds an OpenAI-compatible
client pointed at Ollama, and runs either a one-shot prompt or a REPL.
Every piece of model output is Rich-escaped before display.
"""

from __future__ import annotations

from typing import Annotated, Any

import typer
from openai import OpenAI
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from gehrman.agent.loop import AgentError, make_client, run_turn, start_session
from gehrman.catalogs.store import db_path
from gehrman.config import AppConfig, load_config

console = Console()


def _make_recorder(seen: list[dict[str, Any]]):
    def _on_event(event_type: str, payload: dict[str, Any]) -> None:
        name = payload.get("name", "?")
        if event_type == "tool_call":
            args = payload.get("arguments", "")
            console.print(f"[dim]→ {escape(name)}({escape(str(args))})[/dim]")
        elif event_type == "tool_error":
            console.print(
                f"[yellow]  ! {escape(name)}: {escape(payload.get('error', ''))}[/yellow]"
            )
        elif event_type == "tool_result":
            result = payload.get("result", {})
            if isinstance(result, dict):
                for program in _extract_programs(result):
                    seen.append(program)
                count = result.get("count")
                if count is not None:
                    console.print(f"[dim]  ← {escape(name)}: {count} result(s)[/dim]")
                    return
            console.print(f"[dim]  ← {escape(name)} ok[/dim]")

    return _on_event


def _extract_programs(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Pull program records out of a tool result (search returns a list, get returns one)."""
    if isinstance(result.get("results"), list):
        return [r for r in result["results"] if isinstance(r, dict) and r.get("url")]
    if result.get("found") and result.get("url"):
        return [result]
    return []


def _links_for_answer(answer: str, programs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Dedupe by (platform, handle); keep only programs the answer actually mentions."""
    lowered = answer.lower()
    picked: dict[tuple[str, str], dict[str, Any]] = {}
    for program in programs:
        key = (program.get("platform", ""), program.get("handle", ""))
        if key in picked:
            continue
        handle = (program.get("handle") or "").lower()
        name = (program.get("name") or "").lower()
        if handle and handle in lowered:
            picked[key] = program
        elif name and name in lowered:
            picked[key] = program
    return list(picked.values())


def chat_command(
    prompt: Annotated[
        str | None,
        typer.Option("--prompt", "-p", help="Single-shot: send this prompt and exit."),
    ] = None,
) -> None:
    """Chat with the Gehrman agent using a local Ollama model."""
    try:
        config = load_config()
    except FileNotFoundError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=1) from exc
    if not db_path().is_file():
        console.print(
            f"[red]No database at {db_path()}. Run `gehrman update` first.[/red]"
        )
        raise typer.Exit(code=1)

    client = make_client(config)
    messages = start_session()

    console.print(
        Panel(
            f"Ollama : {config.agent.base_url} ({config.agent.model})\n"
            f"Store  : {db_path()}\n"
            "Type 'exit' or press Ctrl-D to quit.",
            title="Gehrman chat",
            border_style="green",
        )
    )

    if prompt is not None:
        _ask(client, config, messages, prompt)
        return

    while True:
        try:
            user_text = typer.prompt("you", prompt_suffix="> ")
        except (typer.Abort, KeyboardInterrupt):
            console.print()
            return
        stripped = user_text.strip()
        if not stripped:
            continue
        if stripped.lower() in {"exit", "quit"}:
            return
        _ask(client, config, messages, stripped)


def _ask(
    client: OpenAI, config: AppConfig, messages: list[dict], user_text: str
) -> None:
    messages.append({"role": "user", "content": user_text})
    seen: list[dict[str, Any]] = []
    try:
        answer = run_turn(client, config, messages, on_event=_make_recorder(seen))
    except AgentError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        return
    body = escape(answer or "(empty response)")
    links = _links_for_answer(answer or "", seen)
    if links:
        link_lines = "\n".join(
            f"• [cyan]{escape(p['platform'])}/{escape(p['handle'])}[/cyan] — "
            f"[link={escape(p['url'])}]{escape(p['url'])}[/link]"
            for p in links
        )
        body = f"{body}\n\n[bold]Links[/bold]\n{link_lines}"
    console.print(Panel(body, title="gehrman", border_style="cyan"))
