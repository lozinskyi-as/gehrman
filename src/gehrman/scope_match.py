"""Host-based scope matching for `check_scope`.

The old check tested whether the candidate was a substring of the
program URL. That accepted `example.com.attacker.net` for a program at
`example.com`, which is the opposite of what a scope check is for.

Rules here:

- The candidate and every scope identifier are reduced to a hostname:
  scheme, path, query, port, userinfo and a trailing dot are dropped,
  the result is lower-cased and converted to IDNA (punycode).
- A plain identifier (`example.com`, `https://app.example.com/x`)
  matches only that exact host.
- A wildcard identifier (`*.example.com`) matches any host that ends in
  `.example.com` on a label boundary. It does not match the apex
  `example.com`; platforms list the apex separately when it is in scope.
- Out-of-scope entries are checked first and win over in-scope ones.
- Mobile identifiers (bundle IDs such as `com.example.app`) look like
  hostnames, so assets of type `mobile` are skipped.

Nothing here raises on bad input. An unparseable candidate gives
`decision="invalid_candidate"`.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

_LABEL_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_SKIPPED_TYPES = frozenset({"mobile"})


@dataclass(frozen=True)
class ScopeDecision:
    in_scope: bool
    decision: str
    matched: str | None = None


def normalize_host(value: str) -> str | None:
    """Return the IDNA hostname inside `value`, or None if there is none."""
    text = (value or "").strip()
    if not text:
        return None
    if "://" not in text:
        text = "//" + text
    try:
        host = urlsplit(text).hostname
    except ValueError:
        return None
    if not host:
        return None
    host = host.rstrip(".")
    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError:
        return None
    host = host.lower()
    labels = host.split(".")
    if len(labels) < 2 or not all(_LABEL_RE.match(label) for label in labels):
        return None
    return host


def _pattern(identifier: str) -> tuple[str, bool] | None:
    """Return `(host, is_wildcard)` for a scope identifier, or None."""
    text = (identifier or "").strip().lower()
    wildcard = False
    # Strip a scheme first so `https://*.example.com` is handled too.
    if "://" in text:
        text = text.split("://", 1)[1]
    if text.startswith("*."):
        wildcard = True
        text = text[2:]
    host = normalize_host(text)
    if host is None:
        return None
    return host, wildcard


def _matches(host: str, pattern: tuple[str, bool]) -> bool:
    base, wildcard = pattern
    if wildcard:
        return host.endswith("." + base)
    return host == base


def _first_match(host: str, assets: Iterable[Any]) -> str | None:
    for asset in assets:
        identifier, asset_type = _asset_fields(asset)
        if identifier is None or asset_type in _SKIPPED_TYPES:
            continue
        pattern = _pattern(identifier)
        if pattern is not None and _matches(host, pattern):
            return identifier
    return None


def _asset_fields(asset: Any) -> tuple[str | None, str | None]:
    if isinstance(asset, dict):
        identifier = asset.get("identifier")
        asset_type = asset.get("type")
    else:
        identifier = getattr(asset, "identifier", None)
        asset_type = getattr(asset, "type", None)
    if not isinstance(identifier, str):
        return None, None
    return identifier, (str(asset_type) if asset_type is not None else None)


def evaluate(candidate: str, in_scope: Iterable[Any], out_of_scope: Iterable[Any]) -> ScopeDecision:
    """Decide whether `candidate` is in scope. Out-of-scope wins."""
    host = normalize_host(candidate)
    if host is None:
        return ScopeDecision(in_scope=False, decision="invalid_candidate")
    excluded = _first_match(host, out_of_scope)
    if excluded is not None:
        return ScopeDecision(in_scope=False, decision="out_of_scope_match", matched=excluded)
    included = _first_match(host, in_scope)
    if included is not None:
        return ScopeDecision(in_scope=True, decision="in_scope_match", matched=included)
    return ScopeDecision(in_scope=False, decision="no_match")
