"""Supervisor / router.

Two implementations:

- ``SupervisorAgent`` (deterministic): trivial linear policy that advances
  through Researcher → Analyst → Writer → done by inspecting which fields of
  the state are populated. Cheap, predictable, easy to test.
- ``LLMSupervisorAgent``: asks the model which step to take next, allowing
  re-research when the analysis flags low confidence or significant gaps.
  Falls back to the deterministic policy on any parse failure or exception so
  the workflow never deadlocks on a malformed model output.
"""

from __future__ import annotations

import json
import logging
import re

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

log = logging.getLogger(__name__)

ROUTE_RESEARCHER = "researcher"
ROUTE_ANALYST = "analyst"
ROUTE_WRITER = "writer"
ROUTE_CRITIC = "critic"
ROUTE_DONE = "done"

_VALID_ROUTES = {ROUTE_RESEARCHER, ROUTE_ANALYST, ROUTE_WRITER, ROUTE_CRITIC, ROUTE_DONE}


class SupervisorAgent(BaseAgent):
    name = "supervisor"

    def run(self, state: ResearchState) -> ResearchState:
        next_route = self.decide(state)
        state.record_route(next_route)
        return state

    @staticmethod
    def decide(state: ResearchState) -> str:
        if state.final_answer is not None:
            return ROUTE_DONE
        if state.research_notes is None:
            return ROUTE_RESEARCHER
        if state.analysis_notes is None:
            return ROUTE_ANALYST
        return ROUTE_WRITER


_LLM_SYSTEM = (
    "You are the supervisor of a research pipeline with three workers: "
    "researcher (gathers sources), analyst (extracts findings + gaps + confidence), "
    "writer (composes the final answer with citations). "
    "Given the current state, choose the next worker — or 'done' if the answer is "
    "already produced. Re-running researcher is allowed once if the analyst "
    "reported low confidence or unanswered gaps. Reply ONLY with JSON: "
    '{"route": "...", "reason": "..."} where route is one of '
    "researcher | analyst | writer | done."
)


def _state_summary(state: ResearchState) -> str:
    parts = [
        f"Query: {state.request.query}",
        f"Iteration: {state.iteration} / route_history={state.route_history}",
        f"Has research_notes: {bool(state.research_notes)}",
        f"Has analysis_notes: {bool(state.analysis_notes)}",
        f"Has final_answer: {bool(state.final_answer)}",
    ]
    if state.analysis_notes:
        parts.append(f"Analysis preview: {state.analysis_notes[:400]}")
    if state.errors:
        parts.append(f"Errors: {state.errors[-3:]}")
    return "\n".join(parts)


class LLMSupervisorAgent(SupervisorAgent):
    """Supervisor that delegates the routing decision to an LLM.

    Falls back to ``SupervisorAgent.decide`` on any failure.
    """

    def __init__(self, llm: LLMClient | None = None, max_research_runs: int = 2) -> None:
        self._llm = llm or LLMClient(temperature=0.0)
        self._max_research_runs = max_research_runs

    def run(self, state: ResearchState) -> ResearchState:
        next_route = self._decide_with_llm(state)
        state.record_route(next_route)
        return state

    def _decide_with_llm(self, state: ResearchState) -> str:
        deterministic = SupervisorAgent.decide(state)
        if deterministic in {ROUTE_DONE, ROUTE_RESEARCHER}:
            # No interesting decision yet — skip the LLM call.
            return deterministic

        try:
            resp = self._llm.complete(_LLM_SYSTEM, _state_summary(state), max_tokens=80)
            match = re.search(r"\{.*\}", resp.content, re.DOTALL)
            if not match:
                return deterministic
            parsed = json.loads(match.group(0))
            route = str(parsed.get("route", "")).strip().lower()
            if route not in _VALID_ROUTES:
                return deterministic

            researcher_runs = state.route_history.count(ROUTE_RESEARCHER)
            if route == ROUTE_RESEARCHER and researcher_runs >= self._max_research_runs:
                return ROUTE_WRITER if state.research_notes else deterministic
            return route
        except Exception as exc:
            log.warning("LLM supervisor failed (%s); falling back to deterministic policy", exc)
            return deterministic
