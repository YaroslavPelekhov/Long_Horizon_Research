"""
Portable ClaimStore — the axis-2 substrate of OLS.

Differences from lmw.claims.ClaimStore (which inspired this):
  - Statements are OPAQUE strings (no hard-coded FACTS list).
  - Claims carry a sub_goal pointer so retraction/audit can scope by sub-goal.
  - Retraction is propagated to dependent claims via a (statement → deps) graph.
  - Re-test scheduling is first-class (next_retest_at_budget per claim).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ols.core.types import Claim


@dataclass
class ClaimStore:
    """Persistent revisable theory state."""

    claims: list[Claim] = field(default_factory=list)
    # dependency edges: claim_idx → list of upstream claim_idx
    deps: dict[int, list[int]] = field(default_factory=dict)
    # re-test schedule: claim_idx → next budget at which to re-test (None = never)
    retest_at: dict[int, float | None] = field(default_factory=dict)

    # -- assert / retract -----------------------------------------------------

    def assert_claim(
        self,
        statement: str,
        confidence: float,
        provenance: list[int],
        budget: float,
        sub_goal: str | None = None,
        depends_on: list[int] | None = None,
        retest_at: float | None = None,
    ) -> int:
        """Append a claim; return its index. Idempotent on (statement, active)."""
        for i, c in enumerate(self.claims):
            if c.statement == statement and c.status == "active":
                return i                                      # already held
        c = Claim(
            statement=statement,
            confidence=confidence,
            provenance=list(provenance),
            budget_stamp=budget,
            sub_goal=sub_goal,
        )
        self.claims.append(c)
        idx = len(self.claims) - 1
        if depends_on:
            self.deps[idx] = list(depends_on)
        if retest_at is not None:
            self.retest_at[idx] = retest_at
        return idx

    def retract(
        self, statement_or_idx: str | int, budget: float, cascade: bool = True
    ) -> list[int]:
        """Retract a claim by statement or index; cascade to dependents.

        Returns indices of all claims newly flagged for revalidation
        (the original + cascaded dependents — both retracted unconditionally;
        callers can re-assert with new provenance).
        """
        targets: list[int] = []
        if isinstance(statement_or_idx, int):
            if 0 <= statement_or_idx < len(self.claims):
                targets.append(statement_or_idx)
        else:
            for i, c in enumerate(self.claims):
                if c.statement == statement_or_idx and c.status == "active":
                    targets.append(i)

        retracted: list[int] = []
        for idx in targets:
            c = self.claims[idx]
            if c.status == "active":
                c.retract(budget)
                retracted.append(idx)

        if cascade:
            # find any active claim whose deps contain a retracted idx
            changed = True
            while changed:
                changed = False
                for i, c in enumerate(self.claims):
                    if c.status != "active":
                        continue
                    if any(d in retracted for d in self.deps.get(i, [])):
                        c.retract(budget)
                        retracted.append(i)
                        changed = True
        return retracted

    # -- queries --------------------------------------------------------------

    def active(self) -> list[Claim]:
        return [c for c in self.claims if c.status == "active"]

    def active_at(self, budget: float) -> list[Claim]:
        out = []
        for c in self.claims:
            if c.budget_stamp <= budget and (
                c.status == "active"
                or (c.retracted_at is not None and c.retracted_at > budget)
            ):
                out.append(c)
        return out

    def claims_under(self, sub_goal: str) -> list[Claim]:
        return [c for c in self.active() if c.sub_goal == sub_goal]

    def due_for_retest(self, budget: float) -> list[int]:
        return [i for i, b in self.retest_at.items()
                if b is not None and budget >= b
                and 0 <= i < len(self.claims)
                and self.claims[i].status == "active"]

    # -- presentation (for inner-agent context) ------------------------------

    def render_for_agent(self, max_items: int = 24) -> str:
        """Compact human/LLM-readable theory state. Active claims only."""
        act = self.active()
        if not act:
            return "(theory state: empty)"
        # newest-first; cap to avoid blowing the inner agent's context
        head = list(reversed(act))[:max_items]
        lines = [f"  - [{c.confidence:.2f}] {c.statement}  (goal={c.sub_goal})"
                 for c in head]
        more = len(act) - len(head)
        suffix = f"\n  ... +{more} older active claims" if more > 0 else ""
        return "Current theory state (most recent {} of {} active):\n{}{}".format(
            len(head), len(act), "\n".join(lines), suffix
        )

    # -- ablations ------------------------------------------------------------

    def wipe(self) -> None:
        """The −mem ablation: destroy the persistent theory. Tests axis-2 value."""
        self.claims.clear()
        self.deps.clear()
        self.retest_at.clear()
