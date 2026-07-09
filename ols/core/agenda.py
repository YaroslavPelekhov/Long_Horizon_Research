"""
AgendaController — axis-1 substrate of OLS.

Holds a priority queue over sub-goals with a lifecycle:

    explore  →  pursue  →  validate  →  done
       ↓          ↓           ↓
    abandon    abandon     abandon

The controller chooses which sub-goal to feed to the inner agent next, and
revises its own ordering as the ClaimStore changes (validated claims under a
goal lower its remaining priority; futile budget raises abandonment pressure).

Crucially: the controller does NOT propose new sub-goals on its own (that
would re-introduce domain knowledge). New sub-goals come from
   - the adapter's initial list_subdomains() partition, and
   - the inner agent emitting "spawn_subgoal" actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from ols.core.types import AgendaItem


class SubGoalStatus(str, Enum):
    EXPLORE = "explore"      # newly added, not yet probed
    PURSUE = "pursue"        # actively being worked on
    VALIDATE = "validate"    # has provisional claims, awaiting re-test
    DONE = "done"            # has validated, non-retracted claims
    ABANDONED = "abandoned"  # futility triggered


@dataclass
class AgendaController:
    items: dict[str, AgendaItem] = field(default_factory=dict)
    status: dict[str, SubGoalStatus] = field(default_factory=dict)
    # log of (budget, snapshot) tuples for axis-1 metrics
    log: list[tuple[float, list[tuple[str, SubGoalStatus]]]] = field(default_factory=list)

    # -- lifecycle ------------------------------------------------------------

    def add(self, item: AgendaItem, budget: float) -> None:
        if item.sub_goal in self.items:
            return                       # idempotent
        self.items[item.sub_goal] = item
        self.status[item.sub_goal] = SubGoalStatus.EXPLORE
        self._snapshot(budget)

    def promote(self, sub_goal: str, to: SubGoalStatus, budget: float) -> None:
        if sub_goal not in self.items:
            return
        prev = self.status[sub_goal]
        if prev == to:
            return
        self.status[sub_goal] = to
        self._snapshot(budget)

    def abandon(self, sub_goal: str, budget: float) -> None:
        self.promote(sub_goal, SubGoalStatus.ABANDONED, budget)

    def mark_done(self, sub_goal: str, budget: float) -> None:
        self.promote(sub_goal, SubGoalStatus.DONE, budget)

    def record_spend(self, sub_goal: str, cost: float) -> None:
        if sub_goal in self.items:
            self.items[sub_goal].budget_spent += cost

    def record_claim(self, sub_goal: str) -> None:
        if sub_goal in self.items:
            self.items[sub_goal].claims_added += 1

    # -- selection ------------------------------------------------------------

    def active(self) -> list[str]:
        """Sub-goals not abandoned and not done."""
        return [g for g, s in self.status.items()
                if s not in (SubGoalStatus.ABANDONED, SubGoalStatus.DONE)]

    def pick_next(self) -> str | None:
        """Choose the highest-priority active sub-goal, or None if none left.

        Priority = expected_info_gain − 0.5 × budget_spent (penalize stagnation).
        Tie-break by status (EXPLORE before PURSUE before VALIDATE) then by
        alphabetical sub_goal id (deterministic).
        """
        active = self.active()
        if not active:
            return None
        rank = {SubGoalStatus.EXPLORE: 0,
                SubGoalStatus.PURSUE: 1,
                SubGoalStatus.VALIDATE: 2}
        return min(
            active,
            key=lambda g: (
                rank.get(self.status[g], 9),
                -(self.items[g].expected_info_gain - 0.5 * self.items[g].budget_spent),
                g,
            ),
        )

    # -- presentation ---------------------------------------------------------

    def render_for_agent(self) -> str:
        if not self.items:
            return "(agenda: empty)"
        lines = []
        for g, it in sorted(self.items.items()):
            lines.append(
                f"  - [{self.status[g].value:>9}] {g}: {it.question} "
                f"(spent={it.budget_spent:.2f}, claims={it.claims_added})"
            )
        return "Current agenda:\n" + "\n".join(lines)

    def axis1_ratio(self) -> float:
        """Fraction of budget that went to sub-goals which yielded ≥1 claim.

        A pure agenda-tracking score (independent of claim correctness — that
        is the integrity axis). Mirrors lmw.scorer.axis1_agenda_ratio.
        """
        total = sum(it.budget_spent for it in self.items.values())
        if total <= 0:
            return 0.0
        productive = sum(it.budget_spent for it in self.items.values()
                         if it.claims_added > 0)
        return productive / total

    # -- internals ------------------------------------------------------------

    def _snapshot(self, budget: float) -> None:
        self.log.append(
            (budget, [(g, self.status[g]) for g in sorted(self.items)])
        )
