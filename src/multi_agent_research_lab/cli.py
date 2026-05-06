"""Command-line entrypoint."""

from collections import Counter
from pathlib import Path
from typing import Annotated

import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from multi_agent_research_lab.agents.baseline import BaselineAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.evaluation.benchmark import (
    BenchmarkResults,
    _metrics,
    _run_baseline,
)
from multi_agent_research_lab.evaluation.report import render_run_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.langsmith_sink import make_langsmith_sink
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.observability.tracing import (
    CompositeTraceSink,
    JsonlTraceWriter,
    TraceSink,
)

app = typer.Typer(help="Multi-Agent Research Lab CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def _build_trace_sink(
    jsonl_path: Path | None,
    langsmith: bool,
) -> tuple[TraceSink, JsonlTraceWriter]:
    jsonl_writer = JsonlTraceWriter(jsonl_path) if jsonl_path else JsonlTraceWriter()
    sinks: list[TraceSink] = [jsonl_writer]
    if langsmith:
        ls = make_langsmith_sink()
        if ls is not None:
            sinks.append(ls)
            console.print("[dim]LangSmith sink active[/dim]")
        else:
            console.print(
                "[yellow]--langsmith requested but no LANGSMITH_API_KEY; "
                "ignoring[/yellow]"
            )
    sink: TraceSink = CompositeTraceSink(sinks) if len(sinks) > 1 else jsonl_writer
    return sink, jsonl_writer


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a single-agent baseline (one OpenAI call, no orchestration)."""

    _init()
    state = ResearchState(request=ResearchQuery(query=query))
    BaselineAgent().run(state)
    console.print(Panel.fit(state.final_answer or "(empty)", title="Single-Agent Baseline"))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
    trace_file: Annotated[
        Path | None,
        typer.Option("--trace-file", help="Write trace JSONL to this path"),
    ] = None,
    llm_routing: Annotated[
        bool,
        typer.Option("--llm-routing/--no-llm-routing", help="Use LLM-based supervisor routing"),
    ] = False,
    critic: Annotated[
        bool,
        typer.Option("--critic/--no-critic", help="Enable Critic + revision loop"),
    ] = False,
    langsmith: Annotated[
        bool,
        typer.Option("--langsmith/--no-langsmith", help="Also publish trace events to LangSmith"),
    ] = False,
) -> None:
    """Run the multi-agent workflow (Supervisor → Researcher → Analyst → Writer)."""

    _init()
    state = ResearchState(request=ResearchQuery(query=query))
    sink, jsonl_writer = _build_trace_sink(trace_file, langsmith)
    try:
        workflow = MultiAgentWorkflow(
            trace_writer=sink,
            use_llm_routing=llm_routing,
            use_critic=critic,
        )
        result = workflow.run(state)
    finally:
        jsonl_writer.close()
    console.print(f"[dim]trace written to {jsonl_writer.path}[/dim]")
    console.print(Panel.fit(result.final_answer or "(empty)", title="Multi-Agent Final Answer"))


@app.command()
def benchmark(
    config_path: Annotated[
        Path,
        typer.Option("--config", help="YAML config with benchmark.queries"),
    ] = Path("configs/lab_default.yaml"),
    out_path: Annotated[
        Path,
        typer.Option("--out", help="Where to write the markdown report"),
    ] = Path("reports/benchmark_report.md"),
    llm_routing: Annotated[
        bool,
        typer.Option("--llm-routing/--no-llm-routing"),
    ] = False,
    critic: Annotated[
        bool,
        typer.Option("--critic/--no-critic"),
    ] = False,
    langsmith: Annotated[
        bool,
        typer.Option("--langsmith/--no-langsmith"),
    ] = False,
) -> None:
    """Run the benchmark over the configured query set and write a markdown report."""

    _init()
    cfg = yaml.safe_load(config_path.read_text())
    queries: list[str] = cfg["benchmark"]["queries"]
    lab_cfg = cfg.get("lab", {})
    console.print(f"[bold]Benchmarking {len(queries)} queries[/bold]")

    traces_dir = out_path.parent / "traces"
    results = _run_with_options(
        queries,
        traces_dir,
        llm_routing,
        critic,
        langsmith,
        max_iterations=lab_cfg.get("max_iterations"),
        timeout_seconds=lab_cfg.get("timeout_seconds"),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_run_report(results), encoding="utf-8")

    table = Table(title="Benchmark summary")
    for col in ["system", "query", "latency", "tokens", "cost", "judge", "cite"]:
        table.add_column(col)
    for r in results.runs:
        table.add_row(
            r.system,
            (r.query[:40] + "...") if len(r.query) > 40 else r.query,
            f"{r.latency_seconds:.1f}s",
            str(r.tokens_total),
            f"${r.cost_usd:.4f}",
            "—" if r.judge_score is None else f"{r.judge_score}/5",
            "✓" if r.has_citations else "—",
        )
    console.print(table)
    if results.pairwise:
        counts = Counter(p.winner for p in results.pairwise)
        console.print(
            f"[bold]Pairwise:[/bold] multi-agent {counts.get('multi-agent', 0)} / "
            f"baseline {counts.get('baseline', 0)} / tie {counts.get('tie', 0)}"
        )
    console.print(f"[green]Report written to {out_path}[/green]")


def _run_with_options(
    queries: list[str],
    traces_dir: Path,
    llm_routing: bool,
    critic: bool,
    langsmith: bool,
    *,
    max_iterations: int | None = None,
    timeout_seconds: int | None = None,
) -> BenchmarkResults:
    """Like benchmark.run_benchmark but with workflow-level options threaded through."""

    import random
    import re
    from time import perf_counter

    from multi_agent_research_lab.evaluation.benchmark import (
        PairwiseResult,
        _judge_pairwise,
    )
    from multi_agent_research_lab.services.llm_client import LLMClient

    traces_dir.mkdir(parents=True, exist_ok=True)
    judge = LLMClient(temperature=0.0)
    rng = random.Random(42)
    runs = []
    pairwise: list[PairwiseResult] = []

    for query in queries:
        b_state, b_lat = _run_baseline(query)
        runs.append(_metrics("baseline", query, b_state, b_lat, judge))

        m_state = ResearchState(request=ResearchQuery(query=query))
        safe = re.sub(r"[^a-z0-9]+", "-", query.lower())[:40].strip("-")
        sink, jsonl_writer = _build_trace_sink(traces_dir / f"multi-{safe}.jsonl", langsmith)
        try:
            workflow = MultiAgentWorkflow(
                trace_writer=sink,
                use_llm_routing=llm_routing,
                use_critic=critic,
                max_iterations=max_iterations,
                timeout_seconds=timeout_seconds,
            )
            started = perf_counter()
            workflow.run(m_state)
            m_lat = perf_counter() - started
        finally:
            jsonl_writer.close()
        runs.append(_metrics("multi-agent", query, m_state, m_lat, judge))

        pairwise.append(
            _judge_pairwise(
                query,
                b_state.final_answer or "",
                m_state.final_answer or "",
                judge,
                rng,
            )
        )

    return BenchmarkResults(runs=runs, pairwise=pairwise)


if __name__ == "__main__":
    app()
