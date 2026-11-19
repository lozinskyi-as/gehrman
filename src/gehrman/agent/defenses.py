"""Layered defenses against indirect prompt injection.

Two kinds of defense live here:

- **Input-side** transforms on untrusted catalog text (`sanitize`,
  `data_marking`) before it reaches the model. These return
  `(transformed_text, fired)` where `fired` is True when the defense
  actually did something observable.
- **Output-side / control-plane** checks (`ai_guard_input`,
  `ai_guard_output`, `scope_check`, `number_check`) that decide whether
  a chunk of text (a tool result or the model's reply) is acceptable.
  These return `(ok, evidence)` — on `ok=False` the loop replaces the
  affected text with a safe stub and records the evidence string.

The AI guards call out to a small classifier model. The caller injects
a `Classifier` callable so the loop builds one from `AiGuardConfig`
while tests can substitute a fake. When no classifier is supplied, the
guards no-op (returning `ok=True`) so the chat command keeps working
even if the classifier backend is unavailable.

Nothing here raises for malformed input: defenses log a reason and
return rather than taking the whole turn down.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Any, TypedDict


class ClassifierVerdict(TypedDict):
    clean: bool
    reason: str


Classifier = Callable[[str], ClassifierVerdict]


# Loose hostname grammar. Requires at least one dot and ASCII labels
# separated by dots. We don't try to match every RFC-valid name — this
# runs against model-generated text, so the tradeoff favours readability
# of what matched over corner-case coverage.
_HOST_RE = re.compile(
    r"\b[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
    r"(?:\.[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?)+\b"
)

# Any run of digits of length >= 2, optionally fractional, optionally
# with currency commas. Single digits (0..9) are too common to gate on.
_NUMBER_RE = re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b|\b\d{2,}(?:\.\d+)?\b")


def sanitize(text: str, *, enabled: bool) -> tuple[str, bool]:
    """Drop ASCII control bytes (except newline and tab).

    `fired` is True when at least one byte was dropped.
    """
    if not enabled:
        return text, False
    cleaned = "".join(ch for ch in text if ch in "\n\t" or ord(ch) >= 32)
    return cleaned, cleaned != text


def data_marking(text: str, *, source: str, enabled: bool) -> tuple[str, bool]:
    """Wrap untrusted text in a data fence the system prompt tells the model to ignore."""
    if not enabled:
        return text, False
    return f"<<DATA source={source}>>\n{text}\n<<END>>", True


def ai_guard_input(
    text: str, *, enabled: bool, classify: Classifier | None = None
) -> tuple[bool, str]:
    """Classifier-side check on untrusted text about to enter the model context.

    Returns `(ok, evidence)`. If disabled, `ok=True` and no reason.
    If enabled without a classifier wired, `ok=True` with an
    evidence string noting the no-op — the loop records this so the
    trace never silently misrepresents what ran.
    """
    if not enabled:
        return True, ""
    if classify is None:
        return True, "no classifier wired"
    try:
        verdict = classify(text)
    except Exception as exc:  # noqa: BLE001
        # A failing classifier must not take the turn down; treat as clean
        # and record the failure so the trace shows it.
        return True, f"classifier error: {exc}"
    return bool(verdict.get("clean", True)), str(verdict.get("reason", ""))


def ai_guard_output(
    text: str, *, enabled: bool, classify: Classifier | None = None
) -> tuple[bool, str]:
    """Classifier-side check on the model's reply before display."""
    return ai_guard_input(text, enabled=enabled, classify=classify)


def _extract_hosts(text: str) -> set[str]:
    return {m.group(0).lower().rstrip(".") for m in _HOST_RE.finditer(text.lower())}


def _extract_numbers(text: str) -> set[str]:
    return {m.group(0).replace(",", "") for m in _NUMBER_RE.finditer(text)}


def scope_check(
    text: str, *, enabled: bool, allowed_hosts: Iterable[str]
) -> tuple[bool, str]:
    """Every hostname in `text` must appear in `allowed_hosts`.

    `allowed_hosts` is built by the loop from strings seen in tool
    results this turn, so the agent cannot introduce a host it never
    retrieved. Comparison is case-insensitive and trailing dots are
    stripped.
    """
    if not enabled:
        return True, ""
    allow = {h.lower().rstrip(".") for h in allowed_hosts}
    found = _extract_hosts(text)
    unknown = sorted(found - allow)
    if not unknown:
        return True, ""
    # Evidence must not echo the unknown values — they may contain an
    # attacker-controlled canary the whole point of the check is to hide.
    return False, f"{len(unknown)} unknown host(s)"


def number_check(
    text: str, *, enabled: bool, allowed_numbers: Iterable[str]
) -> tuple[bool, str]:
    """Every multi-digit number in `text` must have appeared in a tool result this turn."""
    if not enabled:
        return True, ""
    allow = {n.replace(",", "") for n in allowed_numbers}
    found = _extract_numbers(text)
    unknown = sorted(found - allow)
    if not unknown:
        return True, ""
    return False, f"{len(unknown)} unsourced number(s)"


def collect_allowed_from_result(
    result: Any, hosts_out: set[str], numbers_out: set[str]
) -> None:
    """Walk a tool-result value and seed the allow-sets for scope/number checks."""
    if isinstance(result, dict):
        for value in result.values():
            collect_allowed_from_result(value, hosts_out, numbers_out)
    elif isinstance(result, list):
        for item in result:
            collect_allowed_from_result(item, hosts_out, numbers_out)
    elif isinstance(result, bool):
        return
    elif isinstance(result, int | float):
        numbers_out.add(str(result))
    elif isinstance(result, str):
        hosts_out.update(_extract_hosts(result))
        numbers_out.update(_extract_numbers(result))
