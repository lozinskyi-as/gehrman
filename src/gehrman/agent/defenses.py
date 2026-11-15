"""Hook points for the layered defenses used by the thesis experiment.

Each function respects a boolean switch and falls through to a minimal,
obviously-correct baseline when enabled. Stronger implementations land
in later milestones; this file is the contract the rest of the agent
depends on so switching a defense on/off is a single config flip.
"""

from __future__ import annotations


def sanitize(text: str, *, enabled: bool) -> str:
    if not enabled:
        return text
    # Baseline: drop ASCII control bytes except newline/tab.
    return "".join(ch for ch in text if ch in "\n\t" or ord(ch) >= 32)


def data_marking(text: str, *, source: str, enabled: bool) -> str:
    if not enabled:
        return text
    return f"<<DATA source={source}>>\n{text}\n<<END>>"


def ai_guard_input(text: str, *, enabled: bool) -> bool:
    """Classifier-side input guard. Returns True if the text is clean."""
    if not enabled:
        return True
    del text  # scaffold: replace with a classifier call that uses ai_guard.model
    return True


def ai_guard_output(text: str, *, enabled: bool) -> bool:
    """Classifier-side output guard. Returns True if the text is clean."""
    if not enabled:
        return True
    del text  # scaffold
    return True


def scope_check(candidate: str, allowed: list[str], *, enabled: bool) -> bool:
    """Loose substring match. A real implementation handles wildcards per platform."""
    if not enabled:
        return True
    low = candidate.lower()
    return any(a.lower() in low or low in a.lower() for a in allowed)


def number_check(text: str, allowed: set[float], *, enabled: bool) -> bool:
    """Verify every number in the model's reply came from a tool result."""
    if not enabled:
        return True
    del text, allowed  # scaffold
    return True
