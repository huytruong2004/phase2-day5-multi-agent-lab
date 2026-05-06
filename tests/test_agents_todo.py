"""Smoke tests for the supervisor router (Phase 1 onwards)."""

from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.agents.supervisor import (
    ROUTE_ANALYST,
    ROUTE_DONE,
    ROUTE_RESEARCHER,
    ROUTE_WRITER,
)
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


def _state() -> ResearchState:
    return ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))


def test_supervisor_routes_to_researcher_first() -> None:
    assert SupervisorAgent.decide(_state()) == ROUTE_RESEARCHER


def test_supervisor_routes_to_analyst_after_research() -> None:
    state = _state()
    state.research_notes = "some notes"
    assert SupervisorAgent.decide(state) == ROUTE_ANALYST


def test_supervisor_routes_to_writer_after_analysis() -> None:
    state = _state()
    state.research_notes = "some notes"
    state.analysis_notes = "some analysis"
    assert SupervisorAgent.decide(state) == ROUTE_WRITER


def test_supervisor_stops_when_answer_present() -> None:
    state = _state()
    state.final_answer = "done"
    assert SupervisorAgent.decide(state) == ROUTE_DONE


def test_supervisor_run_records_route_and_iteration() -> None:
    state = _state()
    SupervisorAgent().run(state)
    assert state.route_history == [ROUTE_RESEARCHER]
    assert state.iteration == 1
