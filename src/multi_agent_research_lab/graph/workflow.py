"""Multi-agent workflow with tracing, wall-clock timeout, and error containment.

Hand-rolled supervisor loop: the supervisor emits a route, the workflow
dispatches the corresponding worker, and the loop terminates when the
supervisor emits ``done``, the iteration cap is hit, or the wall-clock budget
is exhausted. Critic-driven revision is opt-in and capped at one round trip.
"""

from time import perf_counter

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.agents.critic import VERDICT_REVISE, CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import (
    ROUTE_ANALYST,
    ROUTE_CRITIC,
    ROUTE_DONE,
    ROUTE_RESEARCHER,
    ROUTE_WRITER,
    LLMSupervisorAgent,
    SupervisorAgent,
)
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import LabError
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.observability.tracing import (
    TraceSink,
    trace_agent,
)

_MAX_CONSECUTIVE_FAILURES = 2


class MultiAgentWorkflow:
    def __init__(
        self,
        supervisor: SupervisorAgent | None = None,
        researcher: ResearcherAgent | None = None,
        analyst: AnalystAgent | None = None,
        writer: WriterAgent | None = None,
        critic: CriticAgent | None = None,
        max_iterations: int | None = None,
        timeout_seconds: int | None = None,
        trace_writer: TraceSink | None = None,
        use_llm_routing: bool = False,
        use_critic: bool = False,
        max_revisions: int = 1,
    ) -> None:
        settings = get_settings()
        self.supervisor = supervisor or (
            LLMSupervisorAgent() if use_llm_routing else SupervisorAgent()
        )
        self.researcher = researcher or ResearcherAgent()
        self.analyst = analyst or AnalystAgent()
        self.writer = writer or WriterAgent()
        self.critic = critic or CriticAgent() if use_critic else None
        self.max_iterations = max_iterations or settings.max_iterations
        self.timeout_seconds = timeout_seconds or settings.timeout_seconds
        self.trace_writer = trace_writer
        self.max_revisions = max_revisions

    def _worker_for(self, route: str) -> BaseAgent | None:
        return {
            ROUTE_RESEARCHER: self.researcher,
            ROUTE_ANALYST: self.analyst,
            ROUTE_WRITER: self.writer,
            ROUTE_CRITIC: self.critic,
        }.get(route)

    def _run_worker(self, worker: BaseAgent, state: ResearchState) -> None:
        with trace_agent(
            state,
            worker.name,
            input_summary=state.request.query,
            writer=self.trace_writer,
        ) as event:
            worker.run(state)
            last = state.agent_results[-1] if state.agent_results else None
            if last is not None:
                event.tokens_in = last.metadata.get("input_tokens")
                event.tokens_out = last.metadata.get("output_tokens")
                event.cost_usd = last.metadata.get("cost_usd")
                event.output_summary = last.content[:160]
                event.extra = {
                    k: v
                    for k, v in last.metadata.items()
                    if k not in {"input_tokens", "output_tokens", "cost_usd", "model"}
                }

    def _maybe_request_revision(self, state: ResearchState) -> bool:
        """Run critic; if it asks for revision and budget remains, clear the
        answer so the supervisor will route back to the writer."""

        if self.critic is None or state.final_answer is None:
            return False
        if state.revision_count >= self.max_revisions:
            return False
        try:
            self._run_worker(self.critic, state)
        except Exception as exc:
            state.errors.append(f"critic: {exc}")
            return False
        last = state.agent_results[-1]
        if last.metadata.get("verdict") == VERDICT_REVISE:
            state.final_answer = None
            state.revision_count += 1
            return True
        return False

    def run(self, state: ResearchState) -> ResearchState:
        started = perf_counter()
        last_failed_route: str | None = None
        consecutive_failures = 0

        for _ in range(self.max_iterations):
            if perf_counter() - started > self.timeout_seconds:
                state.errors.append(f"workflow: timeout after {self.timeout_seconds}s")
                break

            with trace_agent(state, self.supervisor.name, writer=self.trace_writer) as event:
                self.supervisor.run(state)
                event.output_summary = f"route={state.route_history[-1]}"

            route = state.route_history[-1]
            if route == ROUTE_DONE:
                if self._maybe_request_revision(state):
                    continue
                return state
            worker = self._worker_for(route)
            if worker is None:
                raise LabError(f"Unknown route: {route}")

            try:
                self._run_worker(worker, state)
                consecutive_failures = 0
                last_failed_route = None
            except Exception as exc:
                state.errors.append(f"{route}: {exc}")
                if last_failed_route == route:
                    consecutive_failures += 1
                else:
                    consecutive_failures = 1
                    last_failed_route = route
                if consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                    state.errors.append(
                        f"workflow: aborting after {consecutive_failures} "
                        f"consecutive failures on {route}"
                    )
                    break
        else:
            state.errors.append(f"workflow: hit max_iterations={self.max_iterations}")

        if state.final_answer is None:
            try:
                self._run_worker(self.writer, state)
            except Exception as exc:
                state.errors.append(f"writer-fallback: {exc}")
                state.final_answer = state.final_answer or "(workflow failed to produce an answer)"
        return state
