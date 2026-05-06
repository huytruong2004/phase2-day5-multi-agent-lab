"""Critic verdict parsing — unit-level, no LLM calls."""

from multi_agent_research_lab.agents.critic import (
    VERDICT_ACCEPT,
    VERDICT_REVISE,
    _parse_verdict,
)


def test_accept_with_no_issues() -> None:
    verdict, issues = _parse_verdict('{"verdict": "accept", "issues": []}')
    assert verdict == VERDICT_ACCEPT
    assert issues == []


def test_revise_with_issues() -> None:
    verdict, issues = _parse_verdict(
        '{"verdict": "revise", "issues": ["unsupported claim", "missing citation"]}'
    )
    assert verdict == VERDICT_REVISE
    assert issues == ["unsupported claim", "missing citation"]


def test_unknown_verdict_falls_back_to_accept() -> None:
    verdict, _ = _parse_verdict('{"verdict": "maybe", "issues": []}')
    assert verdict == VERDICT_ACCEPT


def test_garbage_output_falls_back_to_accept() -> None:
    verdict, issues = _parse_verdict("not json at all")
    assert verdict == VERDICT_ACCEPT
    assert issues == []


def test_extracts_json_from_surrounding_prose() -> None:
    verdict, issues = _parse_verdict(
        'Sure, here is my verdict: {"verdict": "revise", "issues": ["x"]} done.'
    )
    assert verdict == VERDICT_REVISE
    assert issues == ["x"]
