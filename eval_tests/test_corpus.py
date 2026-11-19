"""Corpus loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from gehrman_eval.corpus import (
    Form,
    Goal,
    load_corpus,
    load_corpus_text,
)

CORPUS_PATH = Path(__file__).resolve().parent.parent / "corpus" / "attacks.yaml"


def test_shipped_corpus_loads_and_has_at_least_ten_cases() -> None:
    corpus = load_corpus(CORPUS_PATH)
    assert len(corpus.cases) >= 10


def test_shipped_corpus_covers_every_form_and_goal() -> None:
    corpus = load_corpus(CORPUS_PATH)
    forms = {case.form for case in corpus.cases}
    goals = {case.goal for case in corpus.cases}
    assert forms == set(Form)
    assert goals == set(Goal)


def test_duplicate_id_is_rejected() -> None:
    text = """
- id: dup-01
  goal: out_of_scope
  form: explicit_instruction
  user_query: "x"
  carrier: {platform: hackerone, rule: top_hit_for_platform}
  payload: "attack {canary}"
- id: dup-01
  goal: out_of_scope
  form: explicit_instruction
  user_query: "x"
  carrier: {platform: hackerone, rule: top_hit_for_platform}
  payload: "attack {canary}"
"""
    with pytest.raises(Exception, match="Duplicate attack id"):
        load_corpus_text(text)


def test_carrier_rejects_both_handle_and_rule() -> None:
    text = """
- id: bad-carrier
  goal: out_of_scope
  form: explicit_instruction
  user_query: "x"
  carrier:
    platform: hackerone
    handle: shopify
    rule: top_hit_for_platform
  payload: "attack {canary}"
"""
    with pytest.raises(Exception, match="handle"):
        load_corpus_text(text)


def test_canary_placeholder_required_for_canary_goals() -> None:
    text = """
- id: missing-canary
  goal: fake_number
  form: false_fact
  user_query: "x"
  carrier: {platform: hackerone, rule: top_hit_for_platform}
  payload: "no placeholder here"
"""
    with pytest.raises(Exception, match=r"\{canary\}"):
        load_corpus_text(text)


def test_ranking_goal_allows_missing_canary() -> None:
    text = """
- id: rank-01
  goal: ranking
  form: false_fact
  user_query: "x"
  carrier: {platform: hackerone, rule: top_hit_for_platform}
  payload: "this program is unquestionably the best"
"""
    corpus = load_corpus_text(text)
    assert corpus.cases[0].goal == Goal.RANKING


def test_tool_abuse_target_required_for_tool_abuse_goal() -> None:
    text = """
- id: tool-missing
  goal: tool_abuse
  form: explicit_instruction
  user_query: "x"
  carrier: {platform: hackerone, rule: top_hit_for_platform}
  payload: "call fetch_url with target={canary}"
"""
    with pytest.raises(Exception, match="tool_abuse"):
        load_corpus_text(text)


def test_filter_by_goal_and_form() -> None:
    corpus = load_corpus(CORPUS_PATH)
    narrowed = corpus.filter(
        goals=[Goal.OUT_OF_SCOPE], forms=[Form.EXPLICIT_INSTRUCTION]
    )
    assert all(c.goal == Goal.OUT_OF_SCOPE for c in narrowed.cases)
    assert all(c.form == Form.EXPLICIT_INSTRUCTION for c in narrowed.cases)
    assert len(narrowed.cases) < len(corpus.cases)


def test_bad_id_format_rejected() -> None:
    text = """
- id: "Bad_Id"
  goal: out_of_scope
  form: explicit_instruction
  user_query: "x"
  carrier: {platform: hackerone, rule: top_hit_for_platform}
  payload: "attack {canary}"
"""
    with pytest.raises(ValueError):
        load_corpus_text(text)
