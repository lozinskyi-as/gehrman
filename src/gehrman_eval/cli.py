"""`gehrman-eval` CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from gehrman_eval.corpus import Form, Goal, load_controls, load_corpus
from gehrman_eval.report import write_report
from gehrman_eval.runner import Runner, dry_run_summary
from gehrman_eval.storage import latest_run_dir
from gehrman_eval.suite import load_suite, merge_cli_overrides

app = typer.Typer(
    name="gehrman-eval",
    help="Prompt-injection evaluation harness for Gehrman.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()


@app.command("validate")
def validate_command(
    corpus: Annotated[
        Path,
        typer.Option("--corpus", help="Path to the attack corpus YAML."),
    ] = Path("corpus/attacks.yaml"),
    controls: Annotated[
        Path,
        typer.Option("--controls", help="Path to the control set YAML."),
    ] = Path("corpus/controls.yaml"),
) -> None:
    """Load and validate the attack corpus and control set."""
    try:
        attack_corpus = load_corpus(corpus)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"Loaded {len(attack_corpus.cases)} attack case(s) from {corpus}")
    if controls.is_file():
        try:
            control_set = load_controls(controls)
        except FileNotFoundError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc
        console.print(f"Loaded {len(control_set.cases)} control case(s) from {controls}")
    else:
        console.print(f"[yellow]No control set at {controls} (skipping).[/yellow]")


@app.command("run")
def run_command(
    suite: Annotated[
        Path | None,
        typer.Option("--suite", help="Suite YAML file (defaults to built-in defaults)."),
    ] = None,
    corpus: Annotated[
        Path | None,
        typer.Option("--corpus", help="Override the attack corpus path."),
    ] = None,
    controls: Annotated[
        Path | None,
        typer.Option("--controls", help="Override the control-set path."),
    ] = None,
    no_controls: Annotated[
        bool,
        typer.Option("--no-controls", help="Skip the control-set run."),
    ] = False,
    model: Annotated[
        list[str] | None,
        typer.Option("--model", help="Model spec; repeatable. e.g. ollama:llama3.1"),
    ] = None,
    profile: Annotated[
        list[str] | None,
        typer.Option("--profile", help="Profile name; repeatable."),
    ] = None,
    attack: Annotated[
        list[str] | None,
        typer.Option("--attack", help="Filter by attack id; repeatable."),
    ] = None,
    goal: Annotated[
        list[Goal] | None,
        typer.Option("--goal", case_sensitive=False, help="Filter by goal; repeatable."),
    ] = None,
    form: Annotated[
        list[Form] | None,
        typer.Option("--form", case_sensitive=False, help="Filter by form; repeatable."),
    ] = None,
    repeats: Annotated[int | None, typer.Option("--repeats")] = None,
    seed: Annotated[int | None, typer.Option("--seed")] = None,
    temperature: Annotated[float | None, typer.Option("--temperature")] = None,
    max_trials: Annotated[
        int | None, typer.Option("--max-trials", help="Hard cap on executed trials.")
    ] = None,
    out_dir: Annotated[Path | None, typer.Option("--out-dir")] = None,
    base_db: Annotated[
        Path | None,
        typer.Option("--base-db", help="Base SQLite catalog; defaults to Gehrman's live DB."),
    ] = None,
    ollama_base_url: Annotated[
        str | None,
        typer.Option("--ollama-base-url", help="Override the Ollama endpoint."),
    ] = None,
    resume: Annotated[
        bool, typer.Option("--resume/--no-resume", help="Skip trials already in results.jsonl.")
    ] = False,
    delivery: Annotated[
        str | None,
        typer.Option("--delivery", help="How the attack reaches the model: natural or forced."),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print the trial plan and exit.")
    ] = False,
) -> None:
    """Execute a full run: inject, score, log to results.jsonl."""
    if delivery is not None and delivery not in ("natural", "forced"):
        console.print("[red]--delivery must be 'natural' or 'forced'.[/red]")
        raise typer.Exit(code=1)
    resolved = merge_cli_overrides(
        load_suite(suite),
        models=model,
        profiles=profile,
        attack_ids=attack,
        goals=goal,
        forms=form,
        repeats=repeats,
        seed=seed,
        temperature=temperature,
        max_trials=max_trials,
        corpus=corpus,
        controls=controls,
        run_controls=False if no_controls else None,
        out_dir=out_dir,
        base_db=base_db,
        ollama_base_url=ollama_base_url,
        delivery=delivery,
    )
    try:
        attack_corpus = load_corpus(resolved.corpus)
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    control_set = None
    if resolved.run_controls and resolved.controls.is_file():
        try:
            control_set = load_controls(resolved.controls)
        except FileNotFoundError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc

    if dry_run:
        summary = dry_run_summary(resolved, attack_corpus, controls=control_set)
        console.print(json.dumps(summary, indent=2))
        return

    run_dir = latest_run_dir(resolved.out_dir) if resume else None
    try:
        report = Runner(resolved, resume=resume, run_dir=run_dir, console=console).run(
            attack_corpus, controls=control_set
        )
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(
        f"[green]Run complete.[/green]\n"
        f"attacks  : planned={report.planned} executed={report.executed} "
        f"skipped={report.skipped} failed={report.failed}\n"
        f"controls : planned={report.controls_planned} "
        f"executed={report.controls_executed} "
        f"skipped={report.controls_skipped} "
        f"failed={report.controls_failed}\n"
        f"Run directory: {report.run_dir}"
    )


@app.command("report")
def report_command(
    run_dir: Annotated[Path, typer.Argument(help="Run directory to summarise.")],
    out_md: Annotated[
        Path | None,
        typer.Option("--out", help="Markdown output path (default: RUN_DIR/report.md)."),
    ] = None,
    out_csv: Annotated[
        Path | None,
        typer.Option("--csv", help="CSV output path (default: RUN_DIR/report.csv)."),
    ] = None,
) -> None:
    """Aggregate a run directory into Markdown + CSV reports."""
    if not run_dir.is_dir():
        console.print(f"[red]Run directory not found: {run_dir}[/red]")
        raise typer.Exit(code=1)
    manifest = run_dir / "manifest.json"
    if not manifest.is_file():
        console.print(f"[red]No manifest.json in {run_dir}[/red]")
        raise typer.Exit(code=1)
    md, csv = write_report(run_dir, out_md=out_md, out_csv=out_csv)
    console.print(f"[green]Wrote[/green] {md}\n[green]Wrote[/green] {csv}")
