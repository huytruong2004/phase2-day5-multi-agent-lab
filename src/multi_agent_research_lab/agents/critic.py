"""Critic agent: fact-checks the writer's answer against the research notes.

The critic emits one of two verdicts:

- ``accept`` — the answer is consistent with the notes; workflow proceeds to done.
- ``revise`` — issues found; workflow sends the answer back to the Writer once
  with the critique attached, then accepts whatever the Writer produces.
"""

from __future__ import annotations

import json
import re

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient

_SYSTEM = (
    "You are a fact-checking critic. Compare a draft answer against the research "
    "notes it should be grounded in. Look for: (1) claims not supported by the "
    "notes, (2) missing citations, (3) factual contradictions, (4) significant "
    "omissions. Be strict but fair. "
    'Reply ONLY with JSON: {"verdict": "accept" | "revise", "issues": [..]}. '
    "issues must be a list of short strings (empty if accept)."
)


VERDICT_ACCEPT = "accept"
VERDICT_REVISE = "revise"


class CriticAgent(BaseAgent):
    name = "critic"

    def __init__(self, llm: LLMClient | None = None) -> None:
        self._llm = llm or LLMClient(temperature=0.0)

    def run(self, state: ResearchState) -> ResearchState:
        if not state.final_answer:
            state.errors.append("critic: no final_answer to review")
            return state

        user = (
            f"Query: {state.request.query}\n\n"
            f"Research notes:\n{state.research_notes or '(none)'}\n\n"
            f"Draft answer:\n{state.final_answer}"
        )
        resp = self._llm.complete(_SYSTEM, user, max_tokens=300)

        verdict, issues = _parse_verdict(resp.content)
        critique_text = (
            "ACCEPT" if verdict == VERDICT_ACCEPT
            else "REVISE — issues: " + "; ".join(issues)
        )
        state.critique = critique_text

        state.agent_results.append(
            AgentResult(
                agent=AgentName.CRITIC,
                content=critique_text,
                metadata={
                    "model": resp.model,
                    "input_tokens": resp.input_tokens,
                    "output_tokens": resp.output_tokens,
                    "cost_usd": resp.cost_usd,
                    "verdict": verdict,
                    "issue_count": len(issues),
                },
            )
        )
        return state


def _parse_verdict(content: str) -> tuple[str, list[str]]:
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return VERDICT_ACCEPT, []
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return VERDICT_ACCEPT, []
    verdict = str(parsed.get("verdict", VERDICT_ACCEPT)).lower()
    issues = parsed.get("issues") or []
    if not isinstance(issues, list):
        issues = []
    issues = [str(i) for i in issues][:5]
    if verdict not in {VERDICT_ACCEPT, VERDICT_REVISE}:
        verdict = VERDICT_ACCEPT
    return verdict, issues
