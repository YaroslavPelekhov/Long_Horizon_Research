"""
Shared dataclasses for OLS.

Design constraints:
  - statement is an OPAQUE string. OLS does not parse it. The adapter (or its
    ground-truth oracle, if any) may validate it. This is the boundary that
    keeps OLS benchmark-agnostic.
  - provenance is a list of int experiment ids (eids). The adapter assigns eids.
  - budget_stamp / budget unit is adapter-defined (LMW=arbitrary budget units,
    DB=API-calls or wall-time-like quanta). OLS treats it as a scalar.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Claim:
    """One belief the agent currently holds (or held)."""
    statement: str                       # opaque domain string
    confidence: float                    # in [0,1]
    provenance: list[int]                # eids that support it (adapter-issued)
    budget_stamp: float                  # adapter budget spent when asserted
    sub_goal: str | None = None          # which sub-goal motivated this claim
    status: str = "active"               # "active" | "retracted"
    retracted_at: float | None = None    # budget when retracted

    def retract(self, budget: float) -> None:
        self.status = "retracted"
        self.retracted_at = budget


@dataclass
class AgendaItem:
    """One sub-goal in the research agenda."""
    sub_goal: str
    question: str
    rationale: str                       # "self" | "imposed" | "auto-derived"
    expected_info_gain: float = 1.0      # priority weight; higher = pursue sooner
    budget_spent: float = 0.0            # cumulative budget consumed on this goal
    claims_added: int = 0                # validated claims added under this goal


@dataclass
class ActionSpec:
    """Adapter-declared action the inner agent may request."""
    name: str
    arg_schema: dict                     # JSON-schema-ish (free-form for v0.1)
    cost_estimate: float                 # adapter's a-priori cost guess
    description: str


@dataclass
class ExperimentResult:
    """What an adapter returns after executing one action."""
    eid: int                             # adapter-issued unique experiment id
    action: str                          # the action name that produced this
    args: dict                           # the args used
    cost: float                          # actual budget consumed
    raw: Any = None                      # adapter-specific payload (samples, etc.)
    summary: dict = field(default_factory=dict)  # adapter-extracted summary stats


@dataclass
class ClaimVerdict:
    """Optional ground-truth check from the adapter, when oracle is available."""
    true: bool
    confidence: float = 1.0              # adapter's confidence in the verdict
    explanation: str = ""
