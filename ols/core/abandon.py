"""
FutilityDetector — axis-6 substrate of OLS.

A sub-goal is "futile" iff it has consumed more than θ_futile of the total
budget without adding any validated (non-retracted) claim.

This is intentionally domain-agnostic: it uses ONLY (budget_spent, claims_added)
per sub-goal — no knowledge of what a "dead-end cluster" or "dud" looks like.
This matters because OLS must be portable to DiscoveryBench, where the
"sub-goals" are decomposed discovery questions and "futility" means an
analysis path that didn't surface a defensible hypothesis component.

θ_futile is exposed as a hyperparameter for robustness analysis (the abandon
analog of the integrity-weight sweep in lmw/sensitivity_l1.py).
"""

from __future__ import annotations

from dataclasses import dataclass

from ols.core.agenda import AgendaController, SubGoalStatus


@dataclass
class FutilityDetector:
    theta_futile: float = 0.30            # max fraction of total budget per goal
    min_absolute_spend: float = 0.05      # below this, never abandon (warm-up)
    # 0.30 default mirrors lmw scripted-agent abandonment regime; tunable.

    def check(self, agenda: AgendaController, budget_total: float,
              budget_left: float, budget_now: float) -> list[str]:
        """Return list of sub-goals to abandon at the current step.

        budget_now is the adapter's cumulative spend (= budget_total - budget_left).
        Caller is responsible for actually invoking agenda.abandon(...).
        """
        if budget_total <= 0:
            return []
        abandon: list[str] = []
        for g in agenda.active():
            it = agenda.items[g]
            frac = it.budget_spent / budget_total
            if it.budget_spent < self.min_absolute_spend:
                continue
            if frac > self.theta_futile and it.claims_added == 0:
                abandon.append(g)
        return abandon

    def regret(self, agenda: AgendaController, budget_total: float) -> float:
        """Dead-end regret = fraction of budget spent on goals that ended futile.

        Mirrors lmw.scorer.axis6_deadend_regret in spirit. Used as a diagnostic.
        """
        if budget_total <= 0:
            return 0.0
        wasted = 0.0
        for g, it in agenda.items.items():
            if agenda.status.get(g) == SubGoalStatus.ABANDONED \
               and it.claims_added == 0:
                wasted += it.budget_spent
        return wasted / budget_total
