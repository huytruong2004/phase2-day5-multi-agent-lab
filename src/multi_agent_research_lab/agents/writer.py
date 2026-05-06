"""Writer agent: composes the final answer with citations."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

_SYSTEM = (
    "You are a technical writer. Given a query, research notes, and analysis, produce a "
    "well-structured answer for the requested audience. Cite sources inline using their "
    "titles in square brackets. Keep the tone direct. End with a 'Sources' section listing "
    "every cited title."
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
            f"Research notes:\n{state.research_notes}\n\n"
            f"Analysis:\n{analysis}"
            f"{critique_block}"
        )
        resp = self._llm.complete(_SYSTEM, user, max_tokens=900)

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
