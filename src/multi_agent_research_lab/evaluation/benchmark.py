"""Benchmark: run baseline vs multi-agent over a fixed query set, collect metrics.

Two judges:

- Per-answer judge keeps a coarse 0-5 score for each individual run (helpful when
  you want absolute floors, e.g. "did this answer the question at all?").
- Pairwise judge compares the two systems head-to-head on the same query and
  picks a winner with reasoning. This is the discriminating signal — the 0-5
  scale saturates at 5 when both answers are above the floor.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

from multi_agent_research_lab.agents.baseline import BaselineAgent
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.tracing import JsonlTraceWriter
from multi_agent_research_lab.services.llm_client import LLMClient

_JUDGE_SYSTEM = (
    "You are an evaluator. Rate an answer to a query on completeness and accuracy "
    "from 0 to 5 (whole number). Reply with ONLY the digit, nothing else."
)

_PAIRWISE_SYSTEM = (
    "You are an impartial judge comparing two answers to the same query. "
    "Pick the better answer based on: factual accuracy, completeness, citation "
    "quality, and clarity. Length alone is not better. "
    "Reply on two lines:\n"
    "Line 1: exactly one of A, B, or TIE\n"
    "Line 2: one sentence explaining your choice."
)


@dataclass
class RunMetrics:
    system: str  # "baseline" | "multi-agent"
    query: str
    latency_seconds: float
    tokens_total: int
    cost_usd: float
    length_words: int
    has_citations: bool
    judge_score: int | None
    error_count: int
    final_answer: str
    errors: list[str] = field(default_factory=list)


@dataclass
class PairwiseResult:
    query: str
    winner: str  # "baseline" | "multi-agent" | "tie"
    reasoning: str
    a_was_baseline: bool


def _sum_token_metric(state: ResearchState, key: str) -> int:
    return sum(int(r.metadata.get(key) or 0) for r in state.agent_results)


def _sum_cost(state: ResearchState) -> float:
    return sum(float(r.metadata.get("cost_usd") or 0.0) for r in state.agent_results)


_CITATION_PATTERN = re.compile(r"\[(?:[^\]]+)\]|^Sources?\b", re.MULTILINE)


def _has_citations(answer: str) -> bool:
    return bool(_CITATION_PATTERN.search(answer or ""))


def _word_count(text: str) -> int:
    return len((text or "").split())


def _judge_score(query: str, answer: str, judge: LLMClient) -> int | None:
    if not answer.strip():
        return 0
    user = f"Query:\n{query}\n\nAnswer:\n{answer}"
    try:
        resp = judge.complete(_JUDGE_SYSTEM, user, max_tokens=5)
        digits = re.findall(r"[0-5]", resp.content)
        return int(digits[0]) if digits else None
    except Exception:
        return None


def _judge_pairwise(
    query: str,
    baseline_answer: str,
    multi_answer: str,
    judge: LLMClient,
    rng: random.Random,
) -> PairwiseResult:
    """Compare baseline vs multi-agent, randomizing which answer is shown as A
    so the judge cannot rely on position. Returns a winner identified by system."""

    a_is_baseline = rng.random() < 0.5
    a_text, b_text = (
        (baseline_answer, multi_answer) if a_is_baseline else (multi_answer, baseline_answer)
    )
    user = (
        f"Query:\n{query}\n\n"
        f"--- Answer A ---\n{a_text or '(empty)'}\n\n"
        f"--- Answer B ---\n{b_text or '(empty)'}"
    )
    try:
        resp = judge.complete(_PAIRWISE_SYSTEM, user, max_tokens=120)
    except Exception as exc:
        return PairwiseResult(
            query=query,
            winner="tie",
            reasoning=f"judge error: {exc}",
            a_was_baseline=a_is_baseline,
        )

    lines = [line.strip() for line in resp.content.strip().splitlines() if line.strip()]
    verdict = lines[0].upper() if lines else ""
    reasoning = lines[1] if len(lines) > 1 else ""

    if verdict.startswith("A"):
        winner = "baseline" if a_is_baseline else "multi-agent"
    elif verdict.startswith("B"):
        winner = "multi-agent" if a_is_baseline else "baseline"
    else:
        winner = "tie"
    return PairwiseResult(
        query=query,
        winner=winner,
        reasoning=reasoning,
        a_was_baseline=a_is_baseline,
    )


def _run_baseline(query: str) -> tuple[ResearchState, float]:
    state = ResearchState(request=ResearchQuery(query=query))
    started = perf_counter()
    BaselineAgent().run(state)
    return state, perf_counter() - started


def _run_multi(query: str, traces_dir: Path) -> tuple[ResearchState, float]:
    state = ResearchState(request=ResearchQuery(query=query))
    safe = re.sub(r"[^a-z0-9]+", "-", query.lower())[:40].strip("-")
    writer = JsonlTraceWriter(traces_dir / f"multi-{safe}.jsonl")
    try:
        workflow = MultiAgentWorkflow(trace_writer=writer)
        started = perf_counter()
        workflow.run(state)
        latency = perf_counter() - started
    finally:
        writer.close()
    return state, latency


def _metrics(
    system: str,
    query: str,
    state: ResearchState,
    latency: float,
    judge: LLMClient,
) -> RunMetrics:
    answer = state.final_answer or ""
    total_tokens = (
        _sum_token_metric(state, "input_tokens")
        + _sum_token_metric(state, "output_tokens")
    )
    return RunMetrics(
        system=system,
        query=query,
        latency_seconds=latency,
        tokens_total=total_tokens,
        cost_usd=_sum_cost(state),
        length_words=_word_count(answer),
        has_citations=_has_citations(answer),
        judge_score=_judge_score(query, answer, judge),
        error_count=len(state.errors),
        final_answer=answer,
        errors=list(state.errors),
    )


@dataclass
class BenchmarkResults:
    runs: list[RunMetrics]
    pairwise: list[PairwiseResult]


def run_benchmark(
    queries: list[str],
    *,
    runners: dict[str, Callable[[str], tuple[ResearchState, float]]] | None = None,
    traces_dir: Path | None = None,
    seed: int = 42,
) -> BenchmarkResults:
    """Run each query against each system; compute per-run metrics and pairwise judgments."""

    traces_dir = traces_dir or Path("reports") / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    runners = runners or {
        "baseline": lambda q: _run_baseline(q),
        "multi-agent": lambda q: _run_multi(q, traces_dir),
    }
    judge = LLMClient(temperature=0.0)
    rng = random.Random(seed)
    runs: list[RunMetrics] = []
    pairwise: list[PairwiseResult] = []

    for query in queries:
        states_by_system: dict[str, ResearchState] = {}
        for system, runner in runners.items():
            state, latency = runner(query)
            runs.append(_metrics(system, query, state, latency, judge))
            states_by_system[system] = state

        if {"baseline", "multi-agent"} <= states_by_system.keys():
            pairwise.append(
                _judge_pairwise(
                    query,
                    states_by_system["baseline"].final_answer or "",
                    states_by_system["multi-agent"].final_answer or "",
                    judge,
                    rng,
                )
            )

    return BenchmarkResults(runs=runs, pairwise=pairwise)
