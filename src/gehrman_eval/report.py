"""Aggregate a run directory into a Markdown report and a CSV table.

Statistics are kept deliberately simple: Wilson 95% confidence intervals
on binomial proportions (attack success rate, false-positive rate) and
counts. No scipy, no bootstrapping — the point is that every number is
reproducible by eye from `results.jsonl`.

The report distinguishes two rates per (model × profile):

- **model_fooled** — the attack effect showed up in the raw model
  answer or tool trace, before output defenses ran.
- **user_exposed** — the same effect survived every enabled defense
  into the text the user actually sees.

The gap between the two is what the defense stack contributed, and is
the main quantity the thesis reports.
"""

from __future__ import annotations

import csv
import io
import json
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_Z_95 = 1.959963984540054  # two-sided 95% normal quantile


@dataclass(frozen=True)
class Proportion:
    successes: int
    n: int
    rate: float
    lower: float
    upper: float

    def format(self) -> str:
        if self.n == 0:
            return "— (n=0)"
        return f"{self.rate:.2%} [{self.lower:.2%}, {self.upper:.2%}]"


def wilson_interval(successes: int, n: int, z: float = _Z_95) -> Proportion:
    """Wilson score interval for a binomial proportion. Handles n=0."""
    if n == 0:
        return Proportion(0, 0, 0.0, 0.0, 0.0)
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    halfwidth = (z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))) / denom
    return Proportion(
        successes=successes,
        n=n,
        rate=p,
        lower=max(0.0, center - halfwidth),
        upper=min(1.0, center + halfwidth),
    )


@dataclass
class AttackRecord:
    trial_id: str
    attack_id: str
    goal: str
    form: str
    model: str
    profile: str
    model_fooled: bool
    user_exposed: bool
    defenses_fired: list[str]
    user_query: str
    raw_answer: str
    final_answer: str
    canary: str
    tokens_in: int
    tokens_out: int
    latency_s: float
    error: str | None


@dataclass
class ControlRecord:
    trial_id: str
    control_id: str
    model: str
    profile: str
    any_defense_fired: bool
    defenses_fired: list[str]
    user_query: str
    raw_answer: str
    final_answer: str
    latency_s: float
    error: str | None


@dataclass
class RunData:
    run_dir: Path
    manifest: dict[str, Any]
    attacks: list[AttackRecord] = field(default_factory=list)
    controls: list[ControlRecord] = field(default_factory=list)


def load_run(run_dir: Path) -> RunData:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    attacks: list[AttackRecord] = []
    results_path = run_dir / "results.jsonl"
    if results_path.is_file():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            check = r.get("check") or {}
            attacks.append(
                AttackRecord(
                    trial_id=r["trial_id"],
                    attack_id=r["attack_id"],
                    goal=r["goal"],
                    form=r["form"],
                    model=r["model"],
                    profile=r["profile"],
                    model_fooled=bool(check.get("model_fooled")),
                    user_exposed=bool(check.get("user_exposed")),
                    defenses_fired=list(r.get("defenses_fired", [])),
                    user_query=r.get("user_query", ""),
                    raw_answer=r.get("raw_answer", ""),
                    final_answer=r.get("final_answer", ""),
                    canary=r.get("canary", ""),
                    tokens_in=int(r.get("tokens_in", 0) or 0),
                    tokens_out=int(r.get("tokens_out", 0) or 0),
                    latency_s=float(r.get("latency_s", 0.0) or 0.0),
                    error=r.get("error"),
                )
            )
    controls: list[ControlRecord] = []
    controls_path = run_dir / "controls.jsonl"
    if controls_path.is_file():
        for line in controls_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            controls.append(
                ControlRecord(
                    trial_id=r["trial_id"],
                    control_id=r["control_id"],
                    model=r["model"],
                    profile=r["profile"],
                    any_defense_fired=bool(r.get("any_defense_fired")),
                    defenses_fired=list(r.get("defenses_fired", [])),
                    user_query=r.get("user_query", ""),
                    raw_answer=r.get("raw_answer", ""),
                    final_answer=r.get("final_answer", ""),
                    latency_s=float(r.get("latency_s", 0.0) or 0.0),
                    error=r.get("error"),
                )
            )
    return RunData(run_dir=run_dir, manifest=manifest, attacks=attacks, controls=controls)


@dataclass(frozen=True)
class CellStats:
    model: str
    profile: str
    slice_key: str  # "all", or "form=explicit_instruction", etc.
    slice_value: str
    model_fooled: Proportion
    user_exposed: Proportion


def _bucket(records: Iterable[AttackRecord], key) -> dict:
    acc: dict = defaultdict(list)
    for r in records:
        acc[key(r)].append(r)
    return acc


def summarize_attacks(
    records: Iterable[AttackRecord], *, slice_by: str | None = None
) -> list[CellStats]:
    """Compute per-(model × profile) ASR, optionally sliced by `form` or `goal`."""
    records = list(records)
    if slice_by is None:
        buckets = _bucket(records, lambda r: (r.model, r.profile, "all", "all"))
    elif slice_by == "form":
        buckets = _bucket(records, lambda r: (r.model, r.profile, "form", r.form))
    elif slice_by == "goal":
        buckets = _bucket(records, lambda r: (r.model, r.profile, "goal", r.goal))
    else:
        raise ValueError(f"Unknown slice_by: {slice_by!r}")
    out: list[CellStats] = []
    for (model, profile, skey, svalue), rows in sorted(buckets.items()):
        fooled = sum(1 for r in rows if r.model_fooled)
        exposed = sum(1 for r in rows if r.user_exposed)
        n = len(rows)
        out.append(
            CellStats(
                model=model,
                profile=profile,
                slice_key=skey,
                slice_value=svalue,
                model_fooled=wilson_interval(fooled, n),
                user_exposed=wilson_interval(exposed, n),
            )
        )
    return out


@dataclass(frozen=True)
class ControlCellStats:
    model: str
    profile: str
    any_defense_fired: Proportion


def summarize_controls(records: Iterable[ControlRecord]) -> list[ControlCellStats]:
    buckets: dict = defaultdict(list)
    for r in records:
        buckets[(r.model, r.profile)].append(r)
    out: list[ControlCellStats] = []
    for (model, profile), rows in sorted(buckets.items()):
        fired = sum(1 for r in rows if r.any_defense_fired)
        out.append(
            ControlCellStats(
                model=model,
                profile=profile,
                any_defense_fired=wilson_interval(fired, len(rows)),
            )
        )
    return out


def pick_examples(data: RunData, *, max_chars: int = 320) -> list[dict[str, Any]]:
    """Pick up to three illustrative traces: attack landed, attack blocked, FP."""
    examples: list[dict[str, Any]] = []
    attack_landed = next(
        (a for a in data.attacks if a.model_fooled and a.user_exposed), None
    )
    attack_blocked = next(
        (a for a in data.attacks if a.model_fooled and not a.user_exposed), None
    )
    false_positive = next(
        (c for c in data.controls if c.any_defense_fired), None
    )
    if attack_landed is not None:
        examples.append(_format_attack_example("Attack landed", attack_landed, max_chars))
    if attack_blocked is not None:
        examples.append(
            _format_attack_example("Attack blocked by defenses", attack_blocked, max_chars)
        )
    if false_positive is not None:
        examples.append(
            {
                "title": "False positive on clean query",
                "attack_id": "—",
                "control_id": false_positive.control_id,
                "model": false_positive.model,
                "profile": false_positive.profile,
                "fired": false_positive.defenses_fired,
                "user_query": false_positive.user_query[:max_chars],
                "raw": false_positive.raw_answer[:max_chars],
                "final": false_positive.final_answer[:max_chars],
            }
        )
    return examples


def _format_attack_example(
    title: str, a: AttackRecord, max_chars: int
) -> dict[str, Any]:
    return {
        "title": title,
        "attack_id": a.attack_id,
        "goal": a.goal,
        "form": a.form,
        "model": a.model,
        "profile": a.profile,
        "fired": a.defenses_fired,
        "canary": a.canary,
        "user_query": a.user_query[:max_chars],
        "raw": a.raw_answer[:max_chars],
        "final": a.final_answer[:max_chars],
    }


def render_markdown(data: RunData) -> str:
    """Return the Markdown report body."""
    buf = io.StringIO()
    manifest = data.manifest
    buf.write("# Gehrman evaluation report\n\n")
    buf.write(f"- **Run dir**: `{data.run_dir}`\n")
    buf.write(f"- **Started**: {manifest.get('started_at', '—')}\n")
    buf.write(f"- **Gehrman commit**: `{manifest.get('gehrman_git_commit') or '—'}`\n")
    buf.write(f"- **Corpus SHA-256**: `{manifest.get('corpus_sha256', '—')}`\n")
    buf.write(f"- **Base DB SHA-256**: `{manifest.get('base_db_sha256', '—')}`\n")
    buf.write(f"- **Models**: {', '.join(manifest.get('models', []))}\n")
    buf.write(f"- **Profiles**: {', '.join(manifest.get('profiles', []))}\n")
    buf.write(f"- **Repeats**: {manifest.get('repeats', '—')}\n")
    buf.write(f"- **Attack trials**: {len(data.attacks)}\n")
    buf.write(f"- **Control trials**: {len(data.controls)}\n")
    total_s = sum(a.latency_s for a in data.attacks) + sum(
        c.latency_s for c in data.controls
    )
    buf.write(f"- **Model time (sum of trial latencies)**: {total_s:.1f} s\n")
    tokens_in = sum(a.tokens_in for a in data.attacks)
    tokens_out = sum(a.tokens_out for a in data.attacks)
    if tokens_in or tokens_out:
        buf.write(f"- **Tokens**: in={tokens_in:,}, out={tokens_out:,}\n")
    failures = sum(1 for a in data.attacks if a.error) + sum(
        1 for c in data.controls if c.error
    )
    if failures:
        buf.write(f"- **Failed trials**: {failures}\n")
    buf.write("\n## Attack success rate per model × profile\n\n")
    buf.write(_table_attacks(summarize_attacks(data.attacks)))

    buf.write("\n## By attack form\n\n")
    buf.write(_table_attacks(summarize_attacks(data.attacks, slice_by="form")))

    buf.write("\n## By attack goal\n\n")
    buf.write(_table_attacks(summarize_attacks(data.attacks, slice_by="goal")))

    buf.write("\n## False-positive rate on clean controls\n\n")
    buf.write(_table_controls(summarize_controls(data.controls)))

    buf.write("\n## Example traces\n\n")
    for ex in pick_examples(data):
        buf.write(_format_example_md(ex))
    buf.write("\n")
    return buf.getvalue()


def _table_attacks(cells: list[CellStats]) -> str:
    if not cells:
        return "_No attack trials recorded._\n"
    buf = io.StringIO()
    slice_label = cells[0].slice_key if cells[0].slice_key != "all" else None
    header = ["model", "profile"]
    if slice_label:
        header.append(slice_label)
    header += ["n", "model_fooled", "user_exposed", "defense gap"]
    buf.write("| " + " | ".join(header) + " |\n")
    buf.write("|" + "|".join("---" for _ in header) + "|\n")
    for c in cells:
        row = [c.model, c.profile]
        if slice_label:
            row.append(c.slice_value)
        gap = c.model_fooled.rate - c.user_exposed.rate
        row += [
            str(c.model_fooled.n),
            c.model_fooled.format(),
            c.user_exposed.format(),
            f"{gap:+.2%}",
        ]
        buf.write("| " + " | ".join(row) + " |\n")
    return buf.getvalue()


def _table_controls(cells: list[ControlCellStats]) -> str:
    if not cells:
        return "_No control trials recorded._\n"
    buf = io.StringIO()
    buf.write("| model | profile | n | any_defense_fired |\n")
    buf.write("|---|---|---|---|\n")
    for c in cells:
        buf.write(
            f"| {c.model} | {c.profile} | {c.any_defense_fired.n} | "
            f"{c.any_defense_fired.format()} |\n"
        )
    return buf.getvalue()


def _format_example_md(ex: dict[str, Any]) -> str:
    buf = io.StringIO()
    buf.write(f"### {ex['title']}\n\n")
    meta_parts = [f"profile=`{ex['profile']}`", f"model=`{ex['model']}`"]
    if ex.get("attack_id") not in (None, "—"):
        meta_parts.append(f"attack=`{ex['attack_id']}`")
        meta_parts.append(f"goal=`{ex['goal']}` form=`{ex['form']}`")
    if ex.get("control_id"):
        meta_parts.append(f"control=`{ex['control_id']}`")
    if ex.get("canary"):
        meta_parts.append(f"canary=`{ex['canary']}`")
    if ex.get("fired"):
        meta_parts.append(f"fired={ex['fired']}")
    buf.write(" · ".join(meta_parts) + "\n\n")
    buf.write(f"**user**: {ex['user_query']}\n\n")
    buf.write(f"**raw**: {ex['raw']}\n\n")
    buf.write(f"**final**: {ex['final']}\n\n")
    return buf.getvalue()


def render_csv(data: RunData) -> str:
    """CSV with one row per (model, profile, slice) — overall, form, goal, controls."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "kind",
            "model",
            "profile",
            "slice_key",
            "slice_value",
            "n",
            "model_fooled_rate",
            "model_fooled_lower",
            "model_fooled_upper",
            "user_exposed_rate",
            "user_exposed_lower",
            "user_exposed_upper",
            "defense_gap",
        ]
    )
    for slice_by in (None, "form", "goal"):
        for cell in summarize_attacks(data.attacks, slice_by=slice_by):
            gap = cell.model_fooled.rate - cell.user_exposed.rate
            writer.writerow(
                [
                    "attack",
                    cell.model,
                    cell.profile,
                    cell.slice_key,
                    cell.slice_value,
                    cell.model_fooled.n,
                    f"{cell.model_fooled.rate:.4f}",
                    f"{cell.model_fooled.lower:.4f}",
                    f"{cell.model_fooled.upper:.4f}",
                    f"{cell.user_exposed.rate:.4f}",
                    f"{cell.user_exposed.lower:.4f}",
                    f"{cell.user_exposed.upper:.4f}",
                    f"{gap:.4f}",
                ]
            )
    for ccell in summarize_controls(data.controls):
        writer.writerow(
            [
                "control",
                ccell.model,
                ccell.profile,
                "fp",
                "any_defense_fired",
                ccell.any_defense_fired.n,
                f"{ccell.any_defense_fired.rate:.4f}",
                f"{ccell.any_defense_fired.lower:.4f}",
                f"{ccell.any_defense_fired.upper:.4f}",
                "",
                "",
                "",
                "",
            ]
        )
    return buf.getvalue()


def write_report(
    run_dir: Path,
    *,
    out_md: Path | None = None,
    out_csv: Path | None = None,
) -> tuple[Path, Path]:
    data = load_run(run_dir)
    md_path = out_md or (run_dir / "report.md")
    csv_path = out_csv or (run_dir / "report.csv")
    md_path.write_text(render_markdown(data), encoding="utf-8")
    csv_path.write_text(render_csv(data), encoding="utf-8")
    return md_path, csv_path
