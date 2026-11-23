"""Edge cases for host-based scope matching (FR-6)."""

from __future__ import annotations

import pytest

from gehrman.scope_match import evaluate, normalize_host

IN = [
    {"identifier": "example.com", "type": "web"},
    {"identifier": "*.api.example.com", "type": "wildcard"},
    {"identifier": "https://shop.example.org/path", "type": "web"},
    {"identifier": "*.bücher.de", "type": "wildcard"},
    {"identifier": "com.example.app", "type": "mobile"},
]
OUT = [{"identifier": "internal.api.example.com", "type": "web"}]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Example.COM", "example.com"),
        ("https://example.com:8443/x?y=1", "example.com"),
        ("user:pw@example.com", "example.com"),
        ("example.com.", "example.com"),
        ("münchen.de", "xn--mnchen-3ya.de"),
        ("localhost", None),
        ("", None),
        ("not a host", None),
    ],
)
def test_normalize_host(raw: str, expected: str | None) -> None:
    assert normalize_host(raw) == expected


def test_exact_host_matches() -> None:
    assert evaluate("https://EXAMPLE.com/login", IN, OUT).in_scope is True


def test_suffix_attack_does_not_match() -> None:
    decision = evaluate("example.com.attacker.net", IN, OUT)
    assert decision.in_scope is False
    assert decision.decision == "no_match"


def test_prefix_attack_does_not_match() -> None:
    assert evaluate("evilexample.com", IN, OUT).in_scope is False


def test_plain_entry_does_not_cover_subdomains() -> None:
    assert evaluate("www.example.com", IN, OUT).in_scope is False


def test_wildcard_matches_subdomain_on_label_boundary() -> None:
    assert evaluate("v2.api.example.com", IN, OUT).in_scope is True
    assert evaluate("a.b.api.example.com", IN, OUT).in_scope is True
    assert evaluate("xapi.example.com", IN, OUT).in_scope is False


def test_wildcard_does_not_cover_apex() -> None:
    assert evaluate("api.example.com", IN, OUT).in_scope is False


def test_out_of_scope_wins_over_wildcard() -> None:
    decision = evaluate("internal.api.example.com", IN, OUT)
    assert decision.in_scope is False
    assert decision.decision == "out_of_scope_match"


def test_url_identifier_matches_its_host() -> None:
    assert evaluate("shop.example.org", IN, OUT).in_scope is True


def test_idna_wildcard_matches_unicode_and_punycode() -> None:
    assert evaluate("shop.bücher.de", IN, OUT).in_scope is True
    assert evaluate("shop.xn--bcher-kva.de", IN, OUT).in_scope is True


def test_mobile_bundle_id_is_not_a_host_entry() -> None:
    assert evaluate("com.example.app", IN, OUT).in_scope is False


def test_invalid_candidate() -> None:
    assert evaluate("???", IN, OUT).decision == "invalid_candidate"
