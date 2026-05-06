"""Structured tracing.

One TraceEvent per agent step, written to `state.trace`, optionally to a
JSONL file, and optionally fanned out to a remote sink (e.g. LangSmith).
Provider-agnostic: any object with a ``write(event)`` method works as a sink.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

from multi_agent_research_lab.core.state import ResearchState


@dataclass
class TraceEvent:
    step: int
    agent: str
    started_at: float
    duration_ms: int
    status: str  # "ok" | "error"
    input_summary: str = ""
    output_summary: str = ""
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float | None = None
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class TraceSink(Protocol):
    def write(self, event: TraceEvent) -> None: ...


class JsonlTraceWriter:
    """Append TraceEvents to a JSONL file. Cheap to construct, safe to ignore."""

    def __init__(self, path: Path | None = None) -> None:
        if path is None:
            run_id = uuid.uuid4().hex[:12]
            path = Path("reports") / "traces" / f"{run_id}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._fh = path.open("a", encoding="utf-8")

    def write(self, event: TraceEvent) -> None:
        self._fh.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class CompositeTraceSink:
    """Fan out trace events to multiple sinks; one failing sink doesn't stop the others."""

    def __init__(self, sinks: list[TraceSink]) -> None:
        self._sinks = sinks

    def write(self, event: TraceEvent) -> None:
        import contextlib

        for sink in self._sinks:
            with contextlib.suppress(Exception):
                # Sinks are best-effort; never break the workflow on a trace failure.
                sink.write(event)


def _truncate(text: str | None, limit: int = 160) -> str:
    if not text:
        return ""
    text = text.replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


@contextmanager
def trace_agent(
    state: ResearchState,
    agent_name: str,
    *,
    input_summary: str = "",
    writer: TraceSink | None = None,
) -> Iterator[TraceEvent]:
    """Wrap an agent run; populate the event with timing and outcome."""

    started = perf_counter()
    event = TraceEvent(
        step=len(state.trace) + 1,
        agent=agent_name,
        started_at=started,
        duration_ms=0,
        status="ok",
        input_summary=_truncate(input_summary),
    )
    try:
        yield event
    except Exception as exc:
        event.status = "error"
        event.error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        event.duration_ms = int((perf_counter() - started) * 1000)
        state.trace.append(asdict(event))
        if writer is not None:
            writer.write(event)
