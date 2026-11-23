"""Attack-level statistics: repeat collapsing, stability, paired tests.

Repeats at temperature 0 are not independent samples of the same
Bernoulli trial: the same prompt usually yields the same answer. So
the unit of analysis is the attack, not the trial:

- `collapse_repeats` turns the repeats of one (model, profile, attack)
  into a single outcome by majority vote. Failed trials are dropped
  before voting; an attack whose repeats all failed is dropped from n.
  A tie counts as a success for the attacker (conservative for the
  defense).
- `repeat_stability` reports how many attacks gave different outcomes
  across repeats. That is what the repeats are for.
- `mcnemar_exact` compares two configurations on the same attacks
  (paired design) with the exact binomial version of McNemar's test.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from itertools import combinations
from typing import TYPE_CHECKING

from gehrman_eval.corpus import goal_group

if TYPE_CHECKING:
    from gehrman_eval.report import AttackRecord


def _majority(values: Sequence[bool]) -> bool:
    hits = sum(values)
    return hits * 2 >= len(values)


def collapse_repeats(records: Iterable[AttackRecord]) -> list[AttackRecord]:
    """One record per (model, profile, attack_id); errors excluded."""
    groups: dict[tuple[str, str, str], list[AttackRecord]] = defaultdict(list)
    for r in records:
        if r.error:
            continue
        groups[(r.model, r.profile, r.attack_id)].append(r)
    out: list[AttackRecord] = []
    for key in sorted(groups):
        rows = groups[key]
        out.append(
            replace(
                rows[0],
                trial_id=f"{key[0]}|{key[1]}|{key[2]}",
                model_fooled=_majority([r.model_fooled for r in rows]),
                user_exposed=_majority([r.user_exposed for r in rows]),
            )
        )
    return out


@dataclass(frozen=True)
class StabilityStats:
    model: str
    profile: str
    attacks_with_repeats: int
    discordant_fooled: int
    discordant_exposed: int
    failed_trials: int


def repeat_stability(records: Iterable[AttackRecord]) -> list[StabilityStats]:
    groups: dict[tuple[str, str, str], list[AttackRecord]] = defaultdict(list)
    failed: dict[tuple[str, str], int] = defaultdict(int)
    for r in records:
        if r.error:
            failed[(r.model, r.profile)] += 1
            continue
        groups[(r.model, r.profile, r.attack_id)].append(r)
    cells: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0, 0])
    for (model, profile, _attack), rows in groups.items():
        if len(rows) < 2:
            continue
        cell = cells[(model, profile)]
        cell[0] += 1
        cell[1] += len({r.model_fooled for r in rows}) > 1
        cell[2] += len({r.user_exposed for r in rows}) > 1
    keys = sorted(set(cells) | set(failed))
    return [
        StabilityStats(
            model=m,
            profile=p,
            attacks_with_repeats=cells[(m, p)][0] if (m, p) in cells else 0,
            discordant_fooled=cells[(m, p)][1] if (m, p) in cells else 0,
            discordant_exposed=cells[(m, p)][2] if (m, p) in cells else 0,
            failed_trials=failed.get((m, p), 0),
        )
        for m, p in keys
    ]


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the discordant counts.

    `b` = pairs where only the first configuration had the event,
    `c` = pairs where only the second did. Under H0 each discordant
    pair is a fair coin, so the p-value is a two-sided binomial test
    with n = b + c and p = 0.5. Returns 1.0 when there are no
    discordant pairs.
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


@dataclass(frozen=True)
class PairedComparison:
    metric: str
    model_a: str
    profile_a: str
    model_b: str
    profile_b: str
    group: str
    n_pairs: int
    both: int
    only_a: int
    only_b: int
    neither: int
    p_value: float


PROFILE_ORDER = ("none", "prompt", "deterministic", "all")


def _paired(left: dict[str, bool], right: dict[str, bool]) -> tuple[int, int, int, int, int]:
    common = sorted(set(left) & set(right))
    both = sum(1 for a in common if left[a] and right[a])
    only_a = sum(1 for a in common if left[a] and not right[a])
    only_b = sum(1 for a in common if right[a] and not left[a])
    neither = len(common) - both - only_a - only_b
    return len(common), both, only_a, only_b, neither


def paired_comparisons(
    collapsed: Iterable[AttackRecord],
    *,
    metric: str = "user_exposed",
    group: str = "all",
) -> list[PairedComparison]:
    """McNemar comparisons on the same attacks.

    Within each model: every pair of profiles. Across models: the same
    profile on two models. `group` is `all`, `by_construction` or
    `adaptive`.
    """
    outcome: dict[tuple[str, str], dict[str, bool]] = defaultdict(dict)
    for r in collapsed:
        if group != "all" and goal_group(r.goal) != group:
            continue
        outcome[(r.model, r.profile)][r.attack_id] = bool(getattr(r, metric))
    models = sorted({m for m, _ in outcome})
    profiles = [p for p in PROFILE_ORDER if any((m, p) in outcome for m in models)]
    profiles += sorted({p for _, p in outcome} - set(profiles))
    pairs: list[tuple[tuple[str, str], tuple[str, str]]] = []
    for m in models:
        for pa, pb in combinations(profiles, 2):
            if (m, pa) in outcome and (m, pb) in outcome:
                pairs.append(((m, pa), (m, pb)))
    for p in profiles:
        for ma, mb in combinations(models, 2):
            if (ma, p) in outcome and (mb, p) in outcome:
                pairs.append(((ma, p), (mb, p)))
    out: list[PairedComparison] = []
    for a, b in pairs:
        n, both, only_a, only_b, neither = _paired(outcome[a], outcome[b])
        out.append(
            PairedComparison(
                metric=metric,
                model_a=a[0],
                profile_a=a[1],
                model_b=b[0],
                profile_b=b[1],
                group=group,
                n_pairs=n,
                both=both,
                only_a=only_a,
                only_b=only_b,
                neither=neither,
                p_value=mcnemar_exact(only_a, only_b),
            )
        )
    return out
