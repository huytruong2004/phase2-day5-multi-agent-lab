"""Writer agent: composes the final answer with citations.

The Writer is given the *raw source snippets* in addition to the Researcher's
notes and the Analyst's findings. Earlier benchmarks showed multi-agent losing
to baseline in pairwise judging because the 3-stage compression pipeline
(snippets → notes → analysis → answer) discarded too much detail; passing the
raw snippets to the Writer recovers that detail without skipping the
intermediate agents (Analyst still chooses what to emphasize).
"""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

_SYSTEM = (
    "You are a technical writer. Given a query, research notes, an analyst's "
    "findings, and the raw source snippets, produce a thorough, well-structured "
    "answer for the requested audience. Use the analysis to decide structure and "
    "emphasis; pull specific facts and language from the raw snippets when they "
    "add depth. Cite sources inline using their titles in square brackets. Keep "
    "the tone direct. End with a 'Sources' section listing every cited title."
)


def _format_sources(sources: list[SourceDocument]) -> str:
    if not sources:
        return "(no raw sources captured)"
    return "\n\n".join(
        f"[{i + 1}] {s.title}{f' ({s.url})' if s.url else ''}\n{s.snippet}"
        for i, s in enumerate(sources)
    )


class WriterAgent(BaseAgent):
    name = "writer"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm or LLMClient(temperature=0.4)

    def run(self, state: ResearchState) -> ResearchState:
        if not state.research_notes:
            state.final_answer = "Insufficient research to answer."
            state.errors.append("writer: skipped LLM call — research_notes empty")
            state.agent_results.append(
                AgentResult(
                    agent=AgentName.WRITER,
                    content=state.final_answer,
                    metadata={"skipped": True, "reason": "no research_notes"},
                )
            )
            return state

        analysis = state.analysis_notes or "(no analysis provided)"
        critique_block = (
            f"\n\nPrevious draft was rejected. Address these issues:\n{state.critique}"
            if state.critique and state.revision_count > 0
            else ""
        )
        user = (
            f"Query: {state.request.query}\n"
            f"Audience: {state.request.audience}\n\n"
            f"Research notes (digest):\n{state.research_notes}\n\n"
            f"Analysis:\n{analysis}\n\n"
            f"Raw sources (for direct facts and quotes):\n{_format_sources(state.sources)}"
            f"{critique_block}"
        )
        resp = self._llm.complete(_SYSTEM, user, max_tokens=1200)

        state.final_answer = resp.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=resp.content,
                metadata={
                    "model": resp.model,
                    "input_tokens": resp.input_tokens,
                    "output_tokens": resp.output_tokens,
                    "cost_usd": resp.cost_usd,
                },
            )
        )
        return state
