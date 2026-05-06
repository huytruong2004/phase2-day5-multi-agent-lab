"""Render benchmark results to markdown."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.evaluation.benchmark import (
    BenchmarkResults,
    PairwiseResult,
    RunMetrics,
)


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Legacy renderer for `BenchmarkMetrics`. Kept for the existing test."""

    lines = [
        "# Benchmark Report",
        "",
        "| Run | Latency (s) | Cost (USD) | Quality | Notes |",
        "|---|---:|---:|---:|---|",
    ]
    for item in metrics:
        cost = "" if item.estimated_cost_usd is None else f"{item.estimated_cost_usd:.4f}"
        quality = "" if item.quality_score is None else f"{item.quality_score:.1f}"
        lines.append(
            f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} | {item.notes} |"
        )
    return "\n".join(lines) + "\n"


def _fmt_judge(score: int | None) -> str:
    return "—" if score is None else f"{score}/5"


def _per_run_table(runs: list[RunMetrics]) -> list[str]:
    out = [
        "## Per-run results",
        "",
        "| System | Query | Latency (s) | Tokens | Cost (USD) | Words "
        "| Citations | Judge | Errors |",
        "|---|---|---:|---:|---:|---:|:---:|:---:|---:|",
    ]
    for r in runs:
        q_short = r.query if len(r.query) <= 60 else r.query[:57] + "..."
        cite = "✓" if r.has_citations else "—"
        out.append(
            f"| {r.system} | {q_short} | {r.latency_seconds:.2f} "
            f"| {r.tokens_total} | {r.cost_usd:.4f} | {r.length_words} "
            f"| {cite} | {_fmt_judge(r.judge_score)} | {r.error_count} |"
        )
    out.append("")
    return out


def _aggregate_table(by_system: dict[str, list[RunMetrics]]) -> list[str]:
    out = [
        "## Aggregate (mean across queries)",
        "",
        "| System | Latency (s) | Tokens | Cost (USD) | Words "
        "| Citation rate | Judge mean | Errors |",
        "|---|---:|---:|---:|---:|:---:|---:|---:|",
    ]
    for system, rs in by_system.items():
        judge_scores = [r.judge_score for r in rs if r.judge_score is not None]
        cite_rate = sum(1 for r in rs if r.has_citations) / len(rs)
        judge_mean = f"{mean(judge_scores):.2f}" if judge_scores else "—"
        out.append(
            f"| {system} | {mean(r.latency_seconds for r in rs):.2f} | "
            f"{int(mean(r.tokens_total for r in rs))} | {mean(r.cost_usd for r in rs):.4f} | "
            f"{int(mean(r.length_words for r in rs))} | {cite_rate:.0%} | "
            f"{judge_mean} | {sum(r.error_count for r in rs)} |"
        )
    out.append("")
    return out


def _pairwise_section(pairwise: list[PairwiseResult]) -> list[str]:
    if not pairwise:
        return []
    counts = Counter(p.winner for p in pairwise)
    n = len(pairwise)
    out = [
        "## Pairwise judge (head-to-head)",
        "",
        f"Across {n} queries: "
        f"**multi-agent {counts.get('multi-agent', 0)} / "
        f"baseline {counts.get('baseline', 0)} / "
        f"tie {counts.get('tie', 0)}**.",
        "",
        "| Query | Winner | Reasoning |",
        "|---|---|---|",
    ]
    for p in pairwise:
        q_short = p.query if len(p.query) <= 50 else p.query[:47] + "..."
        out.append(f"| {q_short} | **{p.winner}** | {p.reasoning} |")
    out.append("")
    return out


def _analysis_section(
    by_system: dict[str, list[RunMetrics]],
    pairwise: list[PairwiseResult],
) -> list[str]:
    if not ({"baseline", "multi-agent"} <= by_system.keys()):
        return []
    b = by_system["baseline"]
    m = by_system["multi-agent"]
    b_lat, m_lat = mean(r.latency_seconds for r in b), mean(r.latency_seconds for r in m)
    b_cost, m_cost = mean(r.cost_usd for r in b), mean(r.cost_usd for r in m)
    b_cite = sum(1 for r in b if r.has_citations) / len(b)
    m_cite = sum(1 for r in m if r.has_citations) / len(m)

    out = ["## Analysis", ""]
    out.append(
        f"- **Latency:** multi-agent is {m_lat / b_lat:.1f}× the baseline "
        f"({m_lat:.1f}s vs {b_lat:.1f}s) — three sequential LLM calls plus a "
        "search step instead of one."
    )
    out.append(
        f"- **Cost:** multi-agent costs {m_cost / b_cost:.1f}× the baseline per "
        f"query (${m_cost:.4f} vs ${b_cost:.4f})."
    )
    out.append(
        f"- **Citations:** baseline {b_cite:.0%}, multi-agent {m_cite:.0%}. The "
        "multi-agent writer is prompted to cite explicitly and operates on retrieved "
        "snippets; the baseline only has its parametric knowledge."
    )
    if pairwise:
        counts = Counter(p.winner for p in pairwise)
        m_wins = counts.get("multi-agent", 0)
        b_wins = counts.get("baseline", 0)
        ties = counts.get("tie", 0)
        out.append(
            f"- **Pairwise judge:** multi-agent wins {m_wins}/{len(pairwise)}, "
            f"baseline wins {b_wins}/{len(pairwise)}, ties {ties}. "
            "Pairwise discriminates where the 0-5 scale saturated."
        )
    out.append("")
    return out


def _failure_modes_section(runs: list[RunMetrics], pairwise: list[PairwiseResult]) -> list[str]:
    out = ["## Failure modes observed", ""]
    failure_lines = [f"- `{r.system}` on _{r.query[:60]}_: {err}" for r in runs for err in r.errors]
    if failure_lines:
        out.extend(failure_lines)

    judge_scores = [r.judge_score for r in runs if r.judge_score is not None]
    if judge_scores and len(set(judge_scores)) <= 1:
        score = judge_scores[0]
        out.append(
            f"- **Judge saturation:** every run scored {score}/5 on the absolute "
            "scale. The pairwise judge above is what discriminates; the absolute "
            "score is kept as a sanity floor (\"did anyone bomb?\")."
        )

    if pairwise:
        baseline_wins = sum(1 for p in pairwise if p.winner == "baseline")
        multi_wins = sum(1 for p in pairwise if p.winner == "multi-agent")
        if baseline_wins >= len(pairwise) and len(pairwise) > 0:
            out.append(
                "- **Multi-agent loses head-to-head despite higher cost:** baseline "
                f"won all {baseline_wins} pairwise comparisons. The Researcher → "
                "Analyst → Writer pipeline is summarizing the same context three times; "
                "each pass discards material the next one needs. The judge reasoning "
                "consistently cites \"more comprehensive\" and \"more structured\" for "
                "the baseline. **Fixes to investigate:** (1) have the Analyst emit "
                "structured findings the Writer expands rather than summarizes again; "
                "(2) skip the Analyst for short factual queries; (3) raise Writer "
                "`max_tokens` so the final answer can match the baseline's depth; "
                "(4) feed the Writer the raw research snippets, not just notes."
            )
        elif multi_wins == 0 and baseline_wins > 0:
            out.append(
                f"- **Multi-agent never wins pairwise** ({multi_wins}/{len(pairwise)}). "
                "Same root cause as above — diagnose by comparing answer length and "
                "content density."
            )

    short_runs = [r for r in runs if r.length_words and r.length_words < 200]
    if short_runs:
        out.append(
            "- **Short-answer outliers:** "
            + ", ".join(f"`{r.system}`:{r.length_words}w" for r in short_runs)
            + ". **Fix:** raise `max_tokens` or tighten the writer's length contract."
        )

    cite_misses = [r for r in runs if r.system == "multi-agent" and not r.has_citations]
    if cite_misses:
        out.append(
            "- **Missing citations from multi-agent:** writer prompt requires citations "
            "but the regex didn't match. **Fix:** parse a structured `WriterOutput` "
            "schema instead of regex-scanning the answer text."
        )

    if len(out) == 2:  # only the header + blank line
        out.append(
            "- No errors observed and no structural failure modes detected in this run. "
            "The Writer's empty-research fallback path and the workflow's max_iterations "
            "/ timeout guards remain the primary defenses."
        )
    out.append("")
    return out


def _caveats_section() -> list[str]:
    return [
        "## Caveats",
        "",
        "- Pairwise judge uses the same provider as the systems under test. Position "
        "bias is mitigated by randomizing which answer is shown as A, but provider "
        "self-bias remains.",
        "- Each query is run once. Latency, judge verdicts, and live search results "
        "all vary across runs.",
        "- The Researcher uses Tavily when `TAVILY_API_KEY` is set, otherwise a small "
        "mock corpus. Ablating between them shows how much of the multi-agent advantage "
        "comes from retrieval vs orchestration.",
        "",
    ]


def render_run_report(results: BenchmarkResults | list[RunMetrics]) -> str:
    """Render the full benchmark report. Accepts the legacy list-of-runs shape too."""

    if isinstance(results, list):
        runs, pairwise = results, []
    else:
        runs, pairwise = results.runs, results.pairwise

    by_system: dict[str, list[RunMetrics]] = defaultdict(list)
    for r in runs:
        by_system[r.system].append(r)

    lines: list[str] = ["# Multi-Agent vs Single-Agent Benchmark", ""]
    lines += _per_run_table(runs)
    lines += _aggregate_table(by_system)
    lines += _pairwise_section(pairwise)
    lines += _analysis_section(by_system, pairwise)
    lines += _failure_modes_section(runs, pairwise)
    lines += _caveats_section()
    return "\n".join(lines)
