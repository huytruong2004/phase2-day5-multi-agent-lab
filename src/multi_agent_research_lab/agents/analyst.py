"""Analyst agent: turns research notes into structured analysis."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

_SYSTEM = (
    "You are an analyst. Given research notes and the original query, extract: "
    "(1) Key findings (3-6 bullets), (2) Gaps or unanswered questions (1-3 bullets), "
    "(3) A confidence rating from 0.0 to 1.0 with one-line justification. "
    "Be terse. Use this exact structure with markdown headings: "
    "## Key Findings\\n## Gaps\\n## Confidence"
)


class AnalystAgent(BaseAgent):
    name = "analyst"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm or LLMClient(temperature=0.1)

    def run(self, state: ResearchState) -> ResearchState:
        if not state.research_notes:
            from multi_agent_research_lab.core.errors import AgentInputError

            raise AgentInputError("analyst: research_notes missing")
        notes = state.research_notes
        user = f"Query: {state.request.query}\n\nResearch notes:\n{notes}"
        resp = self._llm.complete(_SYSTEM, user, max_tokens=500)

        state.analysis_notes = resp.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
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
