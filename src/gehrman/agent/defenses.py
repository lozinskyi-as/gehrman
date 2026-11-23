"""Layered defenses against indirect prompt injection.

Two kinds of defense live here:

- **Input-side** transforms on untrusted catalog text (`sanitize`,
  `data_marking`) before it reaches the model. These return
  `(transformed_text, fired)` where `fired` is True when the defense
  actually did something observable.
- **Output-side** checks (`scope_check`, `number_check`) that decide
  whether the model's reply is acceptable. These return `(ok, evidence)`.
  On `ok=False` the loop replaces the reply with a safe stub and records
  the evidence string. Evidence never echoes the suspicious value.

Nothing here raises for malformed input: defenses log a reason and
return rather than taking the whole turn down.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from typing import Any

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


# Code points that are invisible or reorder text and carry no meaning in
# a program policy. Attackers use them to hide instructions from human
# reviewers while the model still reads them.
_INVISIBLE_RANGES = (
    (0x00AD, 0x00AD),  # soft hyphen
    (0x200B, 0x200F),  # zero-width space/joiners, LRM/RLM
    (0x202A, 0x202E),  # bidi embeddings and overrides
    (0x2060, 0x2064),  # word joiner, invisible operators
    (0x2066, 0x2069),  # bidi isolates
    (0xFEFF, 0xFEFF),  # zero-width no-break space / BOM
    (0xE0000, 0xE007F),  # Unicode tag characters
)

# Text that imitates the data fence written by `data_marking`. After NFKC
# full-width brackets become ASCII, so one pattern covers both.
_FENCE_RE = re.compile(r"<<\s*/?\s*(?:data|end)\b[^>]*>>?", re.IGNORECASE)
_FENCE_REPLACEMENT = "[marker removed]"

SANITIZE_MAX_CHARS = 4000


def _is_dropped(ch: str) -> bool:
    code = ord(ch)
    if ch in "\n\t":
        return False
    if code < 32 or 0x7F <= code <= 0x9F:
        return True
    return any(lo <= code <= hi for lo, hi in _INVISIBLE_RANGES)


def sanitize(
    text: str, *, enabled: bool, max_chars: int = SANITIZE_MAX_CHARS
) -> tuple[str, bool]:
    """Baseline clean-up of untrusted catalog text.

    In order: NFKC normalization; removal of C0/C1 control characters
    (newline and tab kept), zero-width, bidi-control and Unicode tag
    characters; replacement of anything that imitates the
    `<<DATA ...>>` / `<<END>>` fence; truncation to `max_chars`.

    This is a baseline layer, not a defense against well-formed natural
    language instructions. `fired` is True when the text changed.
    """
    if not enabled:
        return text, False
    cleaned = unicodedata.normalize("NFKC", text)
    cleaned = "".join(ch for ch in cleaned if not _is_dropped(ch))
    cleaned = _FENCE_RE.sub(_FENCE_REPLACEMENT, cleaned)
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars]
    return cleaned, cleaned != text


def data_marking(text: str, *, source: str, enabled: bool) -> tuple[str, bool]:
    """Wrap untrusted text in a data fence the system prompt tells the model to ignore."""
    if not enabled:
        return text, False
    return f"<<DATA source={source}>>\n{text}\n<<END>>", True


def _extract_hosts(text: str) -> set[str]:
    return {m.group(0).lower().rstrip(".") for m in _HOST_RE.finditer(text.lower())}


def _extract_numbers(text: str) -> set[str]:
    return {m.group(0).replace(",", "") for m in _NUMBER_RE.finditer(text)}


def scope_check(text: str, *, enabled: bool, allowed_hosts: Iterable[str]) -> tuple[bool, str]:
    """Every hostname in `text` must appear in `allowed_hosts`.

    `allowed_hosts` is seeded from trusted structured tool fields and
    the user's latest message, not from catalog free text. Comparison
    is case-insensitive and trailing dots are stripped.
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


def number_check(text: str, *, enabled: bool, allowed_numbers: Iterable[str]) -> tuple[bool, str]:
    """Every multi-digit number in `text` must have been seeded this turn."""
    if not enabled:
        return True, ""
    allow = {n.replace(",", "") for n in allowed_numbers}
    found = _extract_numbers(text)
    unknown = sorted(found - allow)
    if not unknown:
        return True, ""
    return False, f"{len(unknown)} unsourced number(s)"


# Structured tool-result keys that may seed scope_check / number_check.
# Free-text catalog fields are absent on purpose:
# - `policy_excerpt` is the injection carrier. `explain` returns it, and
#   the harness writes attack payloads there.
# - `name` is catalog text too. A program title can contain a hostname or
#   a fake figure, so it does not contribute. That is the safe default.
# Any new string field stays untrusted until it is added here. Booleans
# are listed so the walker sees them and ignores them. `results` is only
# the search container.
TRUSTED_ALLOWLIST_KEYS = frozenset(
    {
        "asset_types",
        "bounty_awarded_p",
        "bounty_eligible_share",
        "bounty_max",
        "bounty_min",
        "breadth",
        "count",
        "extras",
        "figures",
        "first_response_p",
        "found",
        "freshness",
        "handle",
        "in_scope",
        "in_scope_count",
        "max_bounty",
        "min_bounty",
        "new_assets_30d",
        "new_program",
        "offers_bounty",
        "out_of_scope_count",
        "pays",
        "platform",
        "report_resolved_p",
        "response_efficiency_p",
        "results",
        "score",
        "score_components",
        "skill_fit",
        "url",
        "value",
        "why",
    }
)


def collect_allowed_from_result(result: Any, hosts_out: set[str], numbers_out: set[str]) -> None:
    """Seed allow-sets from trusted structured fields only.

    Keys outside `TRUSTED_ALLOWLIST_KEYS` are skipped, including
    `policy_excerpt` and `name`, and including text wrapped in a
    data-marking fence. Does not raise.
    """
    if isinstance(result, dict):
        for key, value in result.items():
            if key not in TRUSTED_ALLOWLIST_KEYS:
                continue
            _collect_trusted_value(value, hosts_out, numbers_out)
    elif isinstance(result, list):
        for item in result:
            collect_allowed_from_result(item, hosts_out, numbers_out)


def _collect_trusted_value(value: Any, hosts_out: set[str], numbers_out: set[str]) -> None:
    if isinstance(value, dict):
        collect_allowed_from_result(value, hosts_out, numbers_out)
    elif isinstance(value, list):
        for item in value:
            _collect_trusted_value(item, hosts_out, numbers_out)
    elif isinstance(value, bool):
        return
    elif isinstance(value, int | float):
        _add_number(numbers_out, value)
    elif isinstance(value, str):
        hosts_out.update(_extract_hosts(value))
        numbers_out.update(_extract_numbers(value))


def _add_number(numbers_out: set[str], value: int | float) -> None:
    """Record a structured number in the forms `number_check` extracts.

    SQLite REAL values arrive as floats, so a bounty of 50000 is
    `50000.0`. `str(50000.0)` is `50000.0`, while an answer that says
    50000 yields `50000`. Integral floats are stored as both spellings.
    """
    if isinstance(value, bool):
        return
    if isinstance(value, float) and value.is_integer():
        numbers_out.add(str(int(value)))
    numbers_out.add(str(value))


def collect_allowed_from_latest_user_message(
    messages: Iterable[Any], hosts_out: set[str], numbers_out: set[str]
) -> None:
    """Seed allow-sets from the latest user message.

    The user is trusted. Tool-call arguments are not read here: the
    model writes them and can be steered into repeating a canary.
    """
    latest = ""
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            latest = content
    hosts_out.update(_extract_hosts(latest))
    numbers_out.update(_extract_numbers(latest))
