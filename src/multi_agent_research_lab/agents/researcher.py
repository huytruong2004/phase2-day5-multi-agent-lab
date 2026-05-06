"""Researcher agent: collects sources and writes concise research notes."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient, make_search_client

_SYSTEM = (
    "You are a research assistant. Given a query and a list of sources, produce concise "
    "research notes (5-10 bullets) capturing the most relevant facts, claims, and tensions. "
    "Cite each bullet with the source title in square brackets."
)


class ResearcherAgent(BaseAgent):
    name = "researcher"

    def __init__(
        self,
        llm: LLMClient | None = None,
        search: SearchClient | None = None,
    ) -> None:
        self._llm = llm or LLMClient(temperature=0.2)
        self._search = search or make_search_client()

    def run(self, state: ResearchState) -> ResearchState:
        if not state.request.query.strip():
            from multi_agent_research_lab.core.errors import AgentInputError

            raise AgentInputError("researcher: empty query")
        sources = self._search.search(state.request.query, state.request.max_sources)
        state.sources = sources

        sources_block = "\n".join(
            f"- [{i + 1}] {s.title}: {s.snippet}" for i, s in enumerate(sources)
        )
        user = f"Query: {state.request.query}\n\nSources:\n{sources_block}"
        resp = self._llm.complete(_SYSTEM, user, max_tokens=600)

        state.research_notes = resp.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=resp.content,
                metadata={
                    "model": resp.model,
                    "input_tokens": resp.input_tokens,
                    "output_tokens": resp.output_tokens,
                    "cost_usd": resp.cost_usd,
                    "n_sources": len(sources),
                },
            )
        )
        return state
