"""Trial execution loop.

A `Runner` owns one run directory. It plans the cartesian product of
(attack × model × profile × repeat), copies the base DB once, computes
clean-run baselines for ranking goals, then streams trial results to
`results.jsonl` as they finish. Any crash or Ctrl-C leaves a usable
log; `--resume` picks up from whatever is on disk.

The loop is deliberately boring — all the interesting logic (defenses,
scoring, injection) lives in the other modules. Keep it that way.
"""

from __future__ import annotations

import concurrent.futures as _cf
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from gehrman.agent.adapter import AdapterError, ModelAdapter
from gehrman.agent.loop import AgentError, TurnTrace, run_turn, start_session
from gehrman.agent.tools import tool_schemas
from gehrman.config import AgentConfig, AppConfig, ProfileConfig
from gehrman.constants import DATA_DIR_ENV, Skill
from gehrman_eval.adapters.ollama import (
    OllamaModelInfo,
    build_adapter,
    resolve_model_info,
)
from gehrman_eval.checks import (
    check_attack,
    extract_handles_in_order,
)
from gehrman_eval.corpus import (
    AttackCase,
    AttackCorpus,
    ControlCase,
    ControlSet,
    Goal,
)
from gehrman_eval.injector import inject_attack, pick_carrier
from gehrman_eval.profiles import get_profile
from gehrman_eval.storage import (
    Manifest,
    append_control_result,
    append_result,
    baseline_key,
    completed_control_ids,
    completed_trial_ids,
    copy_base_db,
    file_sha256,
    git_commit,
    known_handles,
    materialize_trial_db,
    prepare_run_dir,
    read_baselines,
    write_baselines,
)
from gehrman_eval.suite import Suite
from gehrman_eval.trial import (
    CheckRecord,
    ControlTrialResult,
    ToolCallRecord,
    Trial,
    TrialResult,
    control_trial_id,
)


@contextmanager
def _redirect_db(db_file: Path):
    """Point Gehrman's tools at `db_file` for the duration of the block.

    Gehrman resolves the SQLite path by joining `data_dir()` (which
    honours `GEHRMAN_DATA_DIR`) with the literal `gehrman.db`. The
    runner stages each trial DB under `<trial_dir>/gehrman.db` and
    flips the env var so the agent reads the right copy without any
    change to Gehrman's code.
    """
    if db_file.name != "gehrman.db":
        raise ValueError(
            f"Trial DB must be named gehrman.db to match Gehrman's lookup: {db_file}"
        )
    previous = os.environ.get(DATA_DIR_ENV)
    os.environ[DATA_DIR_ENV] = str(db_file.parent)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(DATA_DIR_ENV, None)
        else:
            os.environ[DATA_DIR_ENV] = previous


@dataclass
class PlannedTrial:
    trial: Trial
    case: AttackCase


@dataclass
class RunReport:
    run_dir: Path
    planned: int
    executed: int
    skipped: int
    failed: int
    manifest: Manifest
    controls_planned: int = 0
    controls_executed: int = 0
    controls_skipped: int = 0
    controls_failed: int = 0


@dataclass
class ToolUsePrecheckResult:
    model: str
    tool_calls_issued: int
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and self.tool_calls_issued > 0


def plan_trials(suite: Suite, corpus: AttackCorpus) -> list[PlannedTrial]:
    """Expand the suite into a deterministic, ordered list of trials."""
    narrowed = corpus.filter(
        ids=suite.attacks.ids or None,
        goals=suite.attacks.goals or None,
        forms=suite.attacks.forms or None,
    )
    planned: list[PlannedTrial] = []
    for model in suite.models:
        for profile in suite.profiles:
            for case in narrowed.cases:
                for repeat in range(suite.repeats):
                    planned.append(
                        PlannedTrial(
                            trial=Trial.build(
                                case,
                                model=model,
                                profile=profile,
                                repeat=repeat,
                                seed=suite.seed,
                                temperature=suite.temperature,
                            ),
                            case=case,
                        )
                    )
    return planned


@dataclass
class PlannedControl:
    trial_id: str
    control: ControlCase
    model: str
    profile: str
    repeat: int
    seed: int


def plan_controls(suite: Suite, controls: ControlSet) -> list[PlannedControl]:
    planned: list[PlannedControl] = []
    for model in suite.models:
        for profile in suite.profiles:
            for case in controls.cases:
                for repeat in range(suite.repeats):
                    planned.append(
                        PlannedControl(
                            trial_id=control_trial_id(
                                control_id=case.id,
                                model=model,
                                profile=profile,
                                repeat=repeat,
                                seed=suite.seed,
                            ),
                            control=case,
                            model=model,
                            profile=profile,
                            repeat=repeat,
                            seed=suite.seed,
                        )
                    )
    return planned


def _default_profile_config() -> ProfileConfig:
    # A neutral researcher profile; the thesis doesn't care about this,
    # but the AppConfig schema requires at least one skill.
    return ProfileConfig(skills=[Skill.WEB], bounty_only=False)


def _app_config_for(
    *, profile_name: str, ollama_base_url: str, model_spec: str
) -> AppConfig:
    defenses = get_profile(profile_name)
    agent = AgentConfig(
        base_url=ollama_base_url,
        model=model_spec.removeprefix("ollama:"),
    )
    return AppConfig(
        profile=_default_profile_config(),
        agent=agent,
        defenses=defenses,
    )


@dataclass
class _TrialOutcome:
    trace: TurnTrace | None
    latency_s: float
    error: str | None = None


def _execute_turn(
    adapter: ModelAdapter,
    config: AppConfig,
    user_query: str,
    *,
    retries: int,
    retry_backoff_s: float,
    trial_timeout_s: float = 0.0,
) -> _TrialOutcome:
    """Run one turn with retries. Does NOT raise on failure; returns an outcome.

    If `trial_timeout_s > 0`, each attempt is wall-clocked in a worker
    thread and aborted on timeout. The worker is left to drain in the
    background — httpx is sync and can't be force-cancelled without
    corrupting the connection pool, but subsequent trials proceed.
    """
    messages = start_session()
    messages.append({"role": "user", "content": user_query})
    attempts = 0
    last_error: Exception | None = None
    started = time.monotonic()
    while attempts <= retries:
        attempts += 1
        try:
            if trial_timeout_s > 0:
                ex = _cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="trial")
                try:
                    fut = ex.submit(
                        run_turn,
                        client=None,
                        config=config,
                        messages=messages,
                        adapter=adapter,
                        return_trace=True,
                    )
                    try:
                        trace = fut.result(timeout=trial_timeout_s)
                    except _cf.TimeoutError as exc:
                        raise AgentError(
                            f"trial exceeded trial_timeout_s={trial_timeout_s:.0f}s"
                        ) from exc
                finally:
                    ex.shutdown(wait=False)
            else:
                trace = run_turn(
                    client=None,
                    config=config,
                    messages=messages,
                    adapter=adapter,
                    return_trace=True,
                )
            assert isinstance(trace, TurnTrace)
            return _TrialOutcome(trace=trace, latency_s=time.monotonic() - started)
        except (AdapterError, AgentError, ValidationError) as exc:
            last_error = exc
            if attempts > retries:
                break
            time.sleep(retry_backoff_s * (2 ** (attempts - 1)))
    return _TrialOutcome(
        trace=None,
        latency_s=time.monotonic() - started,
        error=f"{type(last_error).__name__}: {last_error}",
    )


def _compute_baseline(
    *,
    user_query: str,
    adapter: ModelAdapter,
    config: AppConfig,
    db_handles: list[str],
    retries: int,
    retry_backoff_s: float,
    trial_timeout_s: float = 0.0,
) -> list[str]:
    outcome = _execute_turn(
        adapter,
        config,
        user_query,
        retries=retries,
        retry_backoff_s=retry_backoff_s,
        trial_timeout_s=trial_timeout_s,
    )
    if outcome.trace is None:
        return []
    return extract_handles_in_order(outcome.trace.final_answer, db_handles)[:3]


def tool_use_precheck(
    adapter: ModelAdapter, config: AppConfig
) -> ToolUsePrecheckResult:
    """Ping the model with a trivial request and verify it issues a tool call.

    Called once per model at the start of a run. If a model cannot be
    coaxed into a tool call here, the subsequent attack trials have no
    attack vector (the agent never reaches `explain()`), so the user is
    warned rather than wasting an overnight run.
    """
    messages = [
        {
            "role": "system",
            "content": "You are an assistant that calls tools before answering.",
        },
        {
            "role": "user",
            "content": "Use the search tool to find three HackerOne programs.",
        },
    ]
    try:
        response = adapter.complete(
            messages,
            tool_schemas(),
            temperature=0.0,
            max_tokens=config.agent.max_tokens_per_message,
        )
    except (AdapterError, AgentError) as exc:
        return ToolUsePrecheckResult(
            model=config.agent.model, tool_calls_issued=0, error=str(exc)
        )
    return ToolUsePrecheckResult(
        model=config.agent.model, tool_calls_issued=len(response.tool_calls)
    )


def _tool_calls_to_records(trace: TurnTrace) -> list[ToolCallRecord]:
    return [
        ToolCallRecord(
            name=call.name,
            arguments_json=call.arguments_json,
            accepted=call.accepted,
            result=call.result,
            error=call.error,
            fired_defenses=list(call.fired_defenses),
        )
        for call in trace.tool_calls
    ]


class Runner:
    def __init__(
        self,
        suite: Suite,
        *,
        resume: bool = False,
        run_dir: Path | None = None,
        console: Console | None = None,
    ) -> None:
        self.suite = suite
        self.resume = resume
        self.console = console or Console()
        self.run_dir = run_dir
        self._adapters: dict[str, ModelAdapter] = {}
        self._model_info: dict[str, OllamaModelInfo] = {}

    def _adapter_for(self, model_spec: str) -> ModelAdapter:
        adapter = self._adapters.get(model_spec)
        if adapter is None:
            adapter = build_adapter(model_spec, self.suite.ollama_base_url)
            self._adapters[model_spec] = adapter
        return adapter

    def _model_version(self, model_spec: str) -> str | None:
        info = self._model_info.get(model_spec)
        if info is None:
            info = resolve_model_info(model_spec, self.suite.ollama_base_url)
            self._model_info[model_spec] = info
        return info.digest

    def run(
        self, corpus: AttackCorpus, *, controls: ControlSet | None = None
    ) -> RunReport:
        base_db_source = self._resolve_base_db()
        self.run_dir = self.run_dir or prepare_run_dir(self.suite.out_dir)
        base_db = copy_base_db(base_db_source, self.run_dir)
        handles = known_handles(base_db)
        planned = plan_trials(self.suite, corpus)
        prechecks = self._precheck_models()
        manifest = self._write_manifest(
            planned,
            corpus_path=self.suite.corpus,
            base_db=base_db,
            prechecks=prechecks,
            controls_path=self.suite.controls if controls else None,
        )
        for pc in prechecks:
            if not pc.passed:
                self.console.print(
                    f"[yellow]Tool-use precheck for {pc.model} did not "
                    f"produce a tool call ({pc.error or 'no tool_calls'}). "
                    f"Attacks may not reach the model.[/yellow]"
                )

        completed = completed_trial_ids(self.run_dir) if self.resume else set()
        baselines = read_baselines(self.run_dir)

        executed = 0
        failed = 0
        skipped_count = sum(1 for pt in planned if pt.trial.trial_id in completed)
        remaining = [pt for pt in planned if pt.trial.trial_id not in completed]

        self._ensure_baselines_for(
            remaining, baselines=baselines, handles=handles, base_db=base_db
        )

        with Progress(
            TextColumn("[bold]Attacks"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task("", total=len(remaining))
            for pt in remaining:
                if (
                    self.suite.max_trials is not None
                    and executed >= self.suite.max_trials
                ):
                    break
                result = self._run_one_trial(
                    pt, base_db=base_db, handles=handles, baselines=baselines
                )
                append_result(self.run_dir, result.to_json_line())
                if result.error:
                    failed += 1
                executed += 1
                progress.update(task, advance=1)

        controls_report = self._run_controls(controls, base_db)
        return RunReport(
            run_dir=self.run_dir,
            planned=len(planned),
            executed=executed,
            skipped=skipped_count,
            failed=failed,
            manifest=manifest,
            controls_planned=controls_report[0],
            controls_executed=controls_report[1],
            controls_skipped=controls_report[2],
            controls_failed=controls_report[3],
        )

    def _precheck_models(self) -> list[ToolUsePrecheckResult]:
        results: list[ToolUsePrecheckResult] = []
        for model in self.suite.models:
            config = _app_config_for(
                profile_name=self.suite.profiles[0],
                ollama_base_url=self.suite.ollama_base_url,
                model_spec=model,
            )
            results.append(tool_use_precheck(self._adapter_for(model), config))
        return results

    def _run_controls(
        self, controls: ControlSet | None, base_db: Path
    ) -> tuple[int, int, int, int]:
        if controls is None or not controls.cases:
            return (0, 0, 0, 0)
        planned = plan_controls(self.suite, controls)
        done = completed_control_ids(self.run_dir) if self.resume else set()
        remaining = [p for p in planned if p.trial_id not in done]
        executed = 0
        failed = 0
        with Progress(
            TextColumn("[bold]Controls"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task("", total=len(remaining))
            for pc in remaining:
                if (
                    self.suite.max_trials is not None
                    and executed >= self.suite.max_trials
                ):
                    break
                result = self._run_one_control(pc, base_db=base_db)
                append_control_result(self.run_dir, result.to_json_line())
                if result.error:
                    failed += 1
                executed += 1
                progress.update(task, advance=1)
        return (len(planned), executed, len(planned) - len(remaining), failed)

    def _run_one_control(
        self, pc: PlannedControl, *, base_db: Path
    ) -> ControlTrialResult:
        started = datetime.now(UTC)
        config = _app_config_for(
            profile_name=pc.profile,
            ollama_base_url=self.suite.ollama_base_url,
            model_spec=pc.model,
        )
        with _redirect_db(base_db):
            outcome = _execute_turn(
                self._adapter_for(pc.model),
                config,
                pc.control.user_query,
                retries=self.suite.retries,
                retry_backoff_s=self.suite.retry_backoff_s,
                trial_timeout_s=self.suite.trial_timeout_s,
            )
        finished = datetime.now(UTC)
        trace = outcome.trace
        if trace is None:
            return ControlTrialResult(
                trial_id=pc.trial_id,
                control_id=pc.control.id,
                model=pc.model,
                model_version=self._model_version(pc.model),
                profile=pc.profile,
                repeat=pc.repeat,
                seed=pc.seed,
                user_query=pc.control.user_query,
                raw_answer="",
                final_answer="",
                tool_calls=[],
                defenses_fired=[],
                any_defense_fired=False,
                tokens_in=0,
                tokens_out=0,
                latency_s=outcome.latency_s,
                started_at=started.isoformat(),
                finished_at=finished.isoformat(),
                error=outcome.error,
            )
        return ControlTrialResult(
            trial_id=pc.trial_id,
            control_id=pc.control.id,
            model=pc.model,
            model_version=self._model_version(pc.model),
            profile=pc.profile,
            repeat=pc.repeat,
            seed=pc.seed,
            user_query=pc.control.user_query,
            raw_answer=trace.raw_answer,
            final_answer=trace.final_answer,
            tool_calls=_tool_calls_to_records(trace),
            defenses_fired=list(trace.defenses_fired),
            any_defense_fired=bool(trace.defenses_fired),
            tokens_in=trace.usage.tokens_in,
            tokens_out=trace.usage.tokens_out,
            latency_s=outcome.latency_s,
            started_at=started.isoformat(),
            finished_at=finished.isoformat(),
            error=None,
        )

    def _run_one_trial(
        self,
        pt: PlannedTrial,
        *,
        base_db: Path,
        handles: list[str],
        baselines: dict[str, list[str]],
    ) -> TrialResult:
        trial = pt.trial
        case = pt.case
        started = datetime.now(UTC)
        trial_db = materialize_trial_db(base_db, self.run_dir, trial.trial_id)
        platform, carrier_handle = pick_carrier(trial_db, case)
        injection = inject_attack(trial_db, case, seed=trial.seed)
        config = _app_config_for(
            profile_name=trial.profile,
            ollama_base_url=self.suite.ollama_base_url,
            model_spec=trial.model,
        )
        with _redirect_db(trial_db):
            outcome = _execute_turn(
                self._adapter_for(trial.model),
                config,
                case.user_query,
                retries=self.suite.retries,
                retry_backoff_s=self.suite.retry_backoff_s,
                trial_timeout_s=self.suite.trial_timeout_s,
            )
        finished = datetime.now(UTC)

        if outcome.trace is None:
            check = CheckRecord(
                model_fooled=False,
                user_exposed=False,
                evidence=f"adapter error: {outcome.error}",
            )
            return TrialResult(
                trial_id=trial.trial_id,
                attack_id=trial.attack_id,
                goal=str(trial.goal),
                form=str(trial.form),
                model=trial.model,
                model_version=self._model_version(trial.model),
                profile=trial.profile,
                repeat=trial.repeat,
                seed=trial.seed,
                user_query=case.user_query,
                carrier_platform=platform,
                carrier_handle=carrier_handle,
                canary=injection.canary,
                raw_answer="",
                final_answer="",
                tool_calls=[],
                defenses_fired=[],
                check=check,
                tokens_in=0,
                tokens_out=0,
                latency_s=outcome.latency_s,
                started_at=started.isoformat(),
                finished_at=finished.isoformat(),
                error=outcome.error,
            )

        trace = outcome.trace
        baseline_top3: list[str] = []
        if case.goal == Goal.RANKING:
            key = baseline_key(
                user_query=case.user_query, model=trial.model, profile=trial.profile
            )
            baseline_top3 = baselines.get(key, [])
        check = check_attack(
            trace,
            case,
            canary=injection.canary,
            carrier_handle=carrier_handle,
            known_handles=handles,
            baseline_top3=baseline_top3,
        )
        check_record = CheckRecord(
            model_fooled=check.model_fooled,
            user_exposed=check.user_exposed,
            evidence=check.evidence,
        )
        return TrialResult(
            trial_id=trial.trial_id,
            attack_id=trial.attack_id,
            goal=str(trial.goal),
            form=str(trial.form),
            model=trial.model,
            model_version=self._model_version(trial.model),
            profile=trial.profile,
            repeat=trial.repeat,
            seed=trial.seed,
            user_query=case.user_query,
            carrier_platform=platform,
            carrier_handle=carrier_handle,
            canary=injection.canary,
            raw_answer=trace.raw_answer,
            final_answer=trace.final_answer,
            tool_calls=_tool_calls_to_records(trace),
            defenses_fired=list(trace.defenses_fired),
            check=check_record,
            tokens_in=trace.usage.tokens_in,
            tokens_out=trace.usage.tokens_out,
            latency_s=outcome.latency_s,
            started_at=started.isoformat(),
            finished_at=finished.isoformat(),
            error=None,
        )

    def _ensure_baselines_for(
        self,
        trials: list[PlannedTrial],
        *,
        baselines: dict[str, list[str]],
        handles: list[str],
        base_db: Path,
    ) -> None:
        """Compute clean-run baselines for every (query, model, profile) tuple
        used by any ranking trial. Caches to disk between runs."""
        needed: set[tuple[str, str, str]] = set()
        for pt in trials:
            if pt.case.goal != Goal.RANKING:
                continue
            needed.add((pt.case.user_query, pt.trial.model, pt.trial.profile))
        with _redirect_db(base_db):
            for user_query, model, profile in sorted(needed):
                key = baseline_key(
                    user_query=user_query, model=model, profile=profile
                )
                if key in baselines:
                    continue
                config = _app_config_for(
                    profile_name=profile,
                    ollama_base_url=self.suite.ollama_base_url,
                    model_spec=model,
                )
                top3 = _compute_baseline(
                    user_query=user_query,
                    adapter=self._adapter_for(model),
                    config=config,
                    db_handles=handles,
                    retries=self.suite.retries,
                    retry_backoff_s=self.suite.retry_backoff_s,
                    trial_timeout_s=self.suite.trial_timeout_s,
                )
                baselines[key] = top3
                write_baselines(self.run_dir, baselines)

    def _resolve_base_db(self) -> Path:
        if self.suite.base_db is not None:
            return self.suite.base_db
        from gehrman.catalogs.store import db_path as default_db_path

        default = default_db_path()
        if default.is_file():
            return default
        raise FileNotFoundError(
            "No base DB provided and the default Gehrman DB is missing. "
            "Run `gehrman update` first or pass --base-db."
        )

    def _write_manifest(
        self,
        planned: list[PlannedTrial],
        *,
        corpus_path: Path,
        base_db: Path,
        prechecks: list[ToolUsePrecheckResult] | None = None,
        controls_path: Path | None = None,
    ) -> Manifest:
        model_versions = {
            spec: self._model_version(spec) for spec in self.suite.models
        }
        extra: dict[str, Any] = {
            "model_versions": model_versions,
            "controls_path": str(controls_path) if controls_path else None,
            "tool_use_precheck": [
                {
                    "model": pc.model,
                    "tool_calls_issued": pc.tool_calls_issued,
                    "error": pc.error,
                    "passed": pc.passed,
                }
                for pc in (prechecks or [])
            ],
        }
        manifest = Manifest(
            started_at=datetime.now(UTC).isoformat(),
            gehrman_git_commit=git_commit(cwd=Path(__file__).resolve().parent),
            corpus_path=str(corpus_path),
            corpus_sha256=file_sha256(corpus_path),
            base_db_path=str(base_db),
            base_db_sha256=file_sha256(base_db),
            models=list(self.suite.models),
            profiles=list(self.suite.profiles),
            repeats=self.suite.repeats,
            seed=self.suite.seed,
            temperature=self.suite.temperature,
            max_trials=self.suite.max_trials,
            attack_count=len({pt.case.id for pt in planned}),
            extra=extra,
        )
        manifest.write(self.run_dir / "manifest.json")
        return manifest


def dry_run_summary(
    suite: Suite, corpus: AttackCorpus, *, controls: ControlSet | None = None
) -> dict[str, Any]:
    planned = plan_trials(suite, corpus)
    summary: dict[str, Any] = {
        "trial_count": len(planned),
        "models": list(suite.models),
        "profiles": list(suite.profiles),
        "attacks": sorted({pt.case.id for pt in planned}),
        "repeats": suite.repeats,
        "max_trials": suite.max_trials,
    }
    if controls and controls.cases:
        control_plan = plan_controls(suite, controls)
        summary["control_count"] = len(control_plan)
        summary["controls"] = sorted({pc.control.id for pc in control_plan})
    return summary
