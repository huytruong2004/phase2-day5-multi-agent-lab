"""Single-agent baseline: one LLM call, no orchestration.

Used as the comparison point for the multi-agent workflow in benchmarks.
"""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

_SYSTEM = (
    "You are a research assistant. Given a query, write a thorough, well-structured answer "
    "for the requested audience. Be specific and direct. If you cite sources, use square "
    "brackets with the source title."
)


class BaselineAgent(BaseAgent):
    name = "baseline"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm or LLMClient(temperature=0.3)

    def run(self, state: ResearchState) -> ResearchState:
        user = f"Query: {state.request.query}\nAudience: {state.request.audience}"
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
                    "baseline": True,
                },
            )
        )
        return state
