"""
Inner-agent contract.

An InnerAgent is a thin object that receives an AgentContext (current sub-goal,
ClaimStore view, budget remaining, env description, available actions) and
emits an AgentResponse (one or more actions to execute, and zero or more
claims to assert or retract).

The contract is intentionally narrow:
  - The InnerAgent does NOT decide which sub-goal to work on next (the
    AgendaController does).
  - The InnerAgent does NOT decide when to abandon (the FutilityDetector does).
  - The InnerAgent does NOT maintain memory of past claims (the ClaimStore does).

What the inner agent IS responsible for: domain-specific reasoning — "given
this sub-question and these previously-validated facts, what action should I
take next, and what should I claim about its result?"

This separation is the heart of OLS. The same overlay should improve a weak
inner LLM more than a strong one (the overlay carries the outer-loop
competence), which is exactly the convergent-validity test the paper sets up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ols.core.types import ActionSpec, Claim


@dataclass
class AgentContext:
    sub_goal: str
    sub_goal_question: str
    env_description: str
    available_actions: list[ActionSpec]
    theory_state_view: str           # ClaimStore.render_for_agent(...)
    agenda_view: str                 # AgendaController.render_for_agent()
    budget_remaining: float
    budget_total: float
    history_for_this_subgoal: list[dict] = field(default_factory=list)
    # (action_name, args, result_summary) for actions executed in THIS sub-goal


@dataclass
class ClaimDelta:
    """One claim-store mutation requested by the inner agent."""
    op: str                          # "assert" | "retract"
    statement: str
    confidence: float = 0.8
    provenance: list[int] = field(default_factory=list)
    depends_on: list[int] = field(default_factory=list)


@dataclass
class ActionRequest:
    """One adapter action the inner agent wants executed."""
    action: str
    args: dict


@dataclass
class AgentResponse:
    actions: list[ActionRequest] = field(default_factory=list)
    claims: list[ClaimDelta] = field(default_factory=list)
    advance_subgoal: bool = False    # if True, mark current sub-goal done
    halt: bool = False               # if True, end the episode
    rationale: str = ""              # free-text for logging / debugging


class InnerAgent(Protocol):
    """The minimal contract."""

    def propose(self, ctx: AgentContext) -> AgentResponse: ...

    # Optional: identifier for logging
    name: str
