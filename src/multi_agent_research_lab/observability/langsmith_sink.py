"""LangSmith trace sink.

Posts each TraceEvent as a flat Run. Hierarchical run trees would need parent
linking and run lifecycle management — overkill for a lab. Anything richer
should use the LangSmith @traceable decorator on the agent methods directly.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.observability.tracing import TraceEvent

log = logging.getLogger(__name__)


class LangSmithSink:
    """Posts a one-off Run per TraceEvent. Best-effort; failures are logged."""

    def __init__(self, project_name: str | None = None) -> None:
        from langsmith import Client  # lazy import; package may not be installed

        settings = get_settings()
        if not settings.langsmith_api_key:
            raise RuntimeError("LANGSMITH_API_KEY not set; cannot create LangSmithSink.")
        self._client = Client(api_key=settings.langsmith_api_key)
        self._project = project_name or settings.langsmith_project

    def write(self, event: TraceEvent) -> None:
        try:
            now = datetime.now(UTC)
            self._client.create_run(
                name=event.agent,
                run_type="chain",
                inputs={"summary": event.input_summary},
                outputs={"summary": event.output_summary} if event.status == "ok" else None,
                error=event.error if event.status == "error" else None,
                start_time=now,
                end_time=now,
                id=uuid.uuid4(),
                project_name=self._project,
                extra={
                    "step": event.step,
                    "duration_ms": event.duration_ms,
                    "tokens_in": event.tokens_in,
                    "tokens_out": event.tokens_out,
                    "cost_usd": event.cost_usd,
                    **event.extra,
                },
            )
        except Exception as exc:
            log.warning("LangSmith write failed (%s); continuing without remote trace", exc)


def make_langsmith_sink() -> LangSmithSink | None:
    """Return a sink if a LangSmith API key is configured, else None."""

    if not get_settings().langsmith_api_key:
        return None
    try:
        return LangSmithSink()
    except Exception as exc:
        log.warning("LangSmith sink unavailable: %s", exc)
        return None
