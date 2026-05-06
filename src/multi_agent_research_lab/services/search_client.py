"""Search clients.

Two implementations behind a common interface: a real Tavily-backed client when
`TAVILY_API_KEY` is set, and a deterministic mock for offline tests and
benchmarks. The factory `make_search_client()` picks one based on settings.
"""

from __future__ import annotations

import logging
from typing import Protocol

from tavily import TavilyClient

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument

log = logging.getLogger(__name__)


class SearchClient(Protocol):
    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]: ...


_MOCK_CORPUS: list[SourceDocument] = [
    SourceDocument(
        title="Building effective agents (Anthropic)",
        url="https://www.anthropic.com/engineering/building-effective-agents",
        snippet=(
            "Effective agents are built by composing simple, well-tested patterns: "
            "prompt chaining, routing, parallelization, orchestrator-workers, and "
            "evaluator-optimizer. Prefer the simplest pattern that solves the problem."
        ),
    ),
    SourceDocument(
        title="LangGraph concepts",
        url="https://langchain-ai.github.io/langgraph/concepts/",
        snippet=(
            "LangGraph models multi-agent workflows as state graphs with typed shared "
            "state, conditional edges, and durable checkpoints. Supervisor patterns "
            "and worker handoffs are first-class."
        ),
    ),
    SourceDocument(
        title="OpenAI Agents SDK orchestration",
        url="https://developers.openai.com/api/docs/guides/agents/orchestration",
        snippet=(
            "Orchestration in the Agents SDK uses handoffs between specialized agents. "
            "Each agent owns a narrow role; the runtime tracks tool calls, retries, and "
            "guardrails like max iterations."
        ),
    ),
    SourceDocument(
        title="GraphRAG: knowledge graphs for retrieval",
        url="https://microsoft.github.io/graphrag/",
        snippet=(
            "GraphRAG augments retrieval-augmented generation with community-detected "
            "knowledge graphs, improving multi-hop reasoning over corpora compared with "
            "flat vector RAG, at the cost of an indexing pipeline."
        ),
    ),
    SourceDocument(
        title="Production guardrails for LLM agents",
        url="https://example.com/agent-guardrails",
        snippet=(
            "Production guardrails include max iteration caps, per-call timeouts, "
            "retry-with-backoff on transient errors, structured output validation, "
            "and cost budgets. Tracing each agent step is non-negotiable for debugging."
        ),
    ),
]


class MockSearchClient:
    """Returns mock SourceDocuments. Deterministic, no network."""

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        q = query.lower()
        scored: list[tuple[int, SourceDocument]] = []
        for doc in _MOCK_CORPUS:
            score = sum(
                1
                for term in q.split()
                if len(term) > 3 and (term in doc.title.lower() or term in doc.snippet.lower())
            )
            scored.append((score, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        ranked = [doc for score, doc in scored if score > 0] or [doc for _, doc in scored]
        return ranked[:max_results]


class TavilySearchClient:
    """Tavily-backed search. Falls back gracefully if the API errors."""

    def __init__(self, api_key: str) -> None:
        self._client = TavilyClient(api_key=api_key)

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        try:
            response = self._client.search(
                query=query,
                max_results=max_results,
                search_depth="basic",
            )
        except Exception as exc:
            log.warning("Tavily search failed: %s; returning empty results", exc)
            return []
        results = response.get("results", []) if isinstance(response, dict) else []
        docs: list[SourceDocument] = []
        for item in results[:max_results]:
            docs.append(
                SourceDocument(
                    title=item.get("title") or item.get("url") or "(untitled)",
                    url=item.get("url"),
                    snippet=(item.get("content") or "")[:600],
                    metadata={"score": item.get("score")},
                )
            )
        return docs


def make_search_client() -> SearchClient:
    """Pick the real client when `TAVILY_API_KEY` is set, else fall back to mock."""

    key = get_settings().tavily_api_key
    if key:
        return TavilySearchClient(key)
    return MockSearchClient()
