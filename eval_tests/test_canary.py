"""Canary token generation and validation."""

from __future__ import annotations

import pytest

from gehrman_eval.canary import (
    RESERVED_TLD_SUFFIXES,
    CanaryKind,
    make_canary,
    validate_host_canary,
)


def test_host_canary_uses_reserved_tld() -> None:
    token = make_canary("oos-01", seed=0, kind=CanaryKind.HOST)
    assert any(token.endswith(suffix) for suffix in RESERVED_TLD_SUFFIXES)


def test_number_canary_is_multi_digit_integer() -> None:
    token = make_canary("num-01", seed=0, kind=CanaryKind.NUMBER)
    assert token.isdigit()
    assert len(token) >= 7


def test_canary_is_deterministic() -> None:
    a = make_canary("case-x", seed=42, kind=CanaryKind.HOST)
    b = make_canary("case-x", seed=42, kind=CanaryKind.HOST)
    assert a == b


def test_canary_varies_with_case_id() -> None:
    a = make_canary("case-a", seed=0, kind=CanaryKind.HOST)
    b = make_canary("case-b", seed=0, kind=CanaryKind.HOST)
    assert a != b


def test_canary_varies_with_seed() -> None:
    a = make_canary("case-x", seed=0, kind=CanaryKind.HOST)
    b = make_canary("case-x", seed=1, kind=CanaryKind.HOST)
    assert a != b


def test_validate_host_canary_rejects_real_tld() -> None:
    with pytest.raises(ValueError, match="reserved"):
        validate_host_canary("staging.attacker.com")


def test_validate_host_canary_accepts_reserved() -> None:
    validate_host_canary("staging-abcd.attacker.example")
    validate_host_canary("x.test")
    validate_host_canary("y.invalid.")
