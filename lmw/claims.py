"""
The instrumentation contract: the typed Claim store and the Agenda.

These are the ONLY structures the harness inspects to compute the
Research Program Score and the axis-1/2/6 metrics. An agent under test must
record every belief it holds as a Claim (with honest provenance) and expose
its current Agenda. The harness can probe correctness, force a re-test, and
audit provenance honesty at any budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Canonical fact statements the world understands (must match SCM.true_edges keys).
# Agents emit these strings; the oracle checks them against ground truth.
FACTS = [
    "confounded(X0,X1,Z)",
    "no_effect(X0,X1)",
    "causal(X2,X1,+)",
    "causal(X2,X1,-)",
    "causal(X0,X1,+)",   # the trap
    "causal(X3,X4,+)",
    "mean(X5)~const",
]


@dataclass
class Claim:
    statement: str                       # one of FACTS
    confidence: float                    # in [0,1]
    provenance: list[int]                # experiment ids that support it
    budget_stamp: float                  # budget spent when first asserted
    status: str = "active"               # "active" | "retracted"
    retracted_at: float | None = None    # budget when retracted

    def retract(self, budget: float) -> None:
        self.status = "retracted"
        self.retracted_at = budget


@dataclass
class AgendaItem:
    subdomain: str            # "A" | "B" | "C"
    question: str
    rationale: str


@dataclass
class ClaimStore:
    """Persistent, revisable theory. The harness reads .claims and .agenda_log."""
    claims: list[Claim] = field(default_factory=list)
    agenda: list[AgendaItem] = field(default_factory=list)
    # log of (budget, agenda snapshot) every revision -- used for axis-1 metrics
    agenda_log: list[tuple[float, list[AgendaItem]]] = field(default_factory=list)

    # -- theory ops -----------------------------------------------------------
    def assert_claim(self, statement: str, confidence: float,
                     provenance: list[int], budget: float) -> Claim:
        c = Claim(statement, confidence, list(provenance), budget)
        self.claims.append(c)
        return c

    def retract(self, statement: str, budget: float) -> None:
        for c in self.claims:
            if c.statement == statement and c.status == "active":
                c.retract(budget)

    def active(self) -> list[Claim]:
        return [c for c in self.claims if c.status == "active"]

    def active_at(self, budget: float) -> list[Claim]:
        """Claims that were active at the given budget point (for the VRK curve)."""
        out = []
        for c in self.claims:
            if c.budget_stamp <= budget and (
                c.status == "active" or (c.retracted_at is not None and c.retracted_at > budget)
            ):
                out.append(c)
        return out

    # -- agenda ops -----------------------------------------------------------
    def set_agenda(self, items: list[AgendaItem], budget: float) -> None:
        self.agenda = list(items)
        self.agenda_log.append((budget, list(items)))

    def wipe(self) -> None:
        """Used by the -mem ablation: destroy the persistent theory."""
        self.claims.clear()
        self.agenda.clear()
        # agenda_log is harness-owned telemetry; keep it for scoring.
