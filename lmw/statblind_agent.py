"""
Statistical-blind baseline (Scripted-blind).

A deterministic agent that operates on the same World surface as the LLM
agents (clusters, observe, intervene, budget, ClaimStore) using ONLY generic
causal-discovery heuristics. It has NO knowledge of LMW invariants — no
notion of "trap cluster", "spurious pair via latent", "dud", "hunt the true
cause among decoys". It correlates, intervenes on strongly correlated
pairs, and claims causal() iff the do-effect is statistically significant.

Purpose: a fair-game comparator for the Pareto-dominance question. The
domain-aware ConnectorAgent (Scripted/Full) is a *process oracle*: a human
who knew the LMW design wrote its strategy. Scripted-blind is what a generic
scripted agent — without LMW domain priors — can do on the same surface.
The gap Scripted/Full → Scripted-blind isolates how much of the scripted
ceiling comes from domain knowledge rather than outer-loop competence.
"""

from __future__ import annotations

import math

from claims import AgendaItem, ClaimStore
from world import BudgetExhausted, World

CORR_THRESH = 0.35      # observed correlation needed to bother testing
EFFECT_Z = 3.0          # |d| > Z*SE to claim causal
N_OBS, N_INT = 8, 6     # sample sizes matching the other agents


def _se_diff(a: list[float], b: list[float]) -> float:
    def var(xs: list[float]) -> float:
        if len(xs) < 2:
            return 1.0
        m = sum(xs) / len(xs)
        return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var(a) / max(1, len(a)) + var(b) / max(1, len(b)))


class StatBlindAgent:
    name = "Scripted-blind"

    def run(self, world: World, store: ClaimStore) -> None:
        clusters = dict(world._scm.structure.subdomains)
        store.set_agenda(
            [AgendaItem(c, "explore", "blind") for c in clusters],
            world.budget_total - world.budget_left)

        # phase 1: observe every cluster, collect strong pairwise correlations
        corrs: list[tuple[float, str, str, int]] = []
        try:
            for cname, vs in sorted(clusters.items()):
                if world.budget_left < N_OBS * 0.5 + 1:
                    break
                o = world.observe(vs, n=N_OBS)
                for i, x in enumerate(vs):
                    for y in vs[i + 1:]:
                        c = abs(o.corr(x, y))
                        if c > CORR_THRESH:
                            corrs.append((c, x, y, o.eid))
            corrs.sort(reverse=True)            # strongest first

            # phase 2: test each strong pair causally; assert causal() iff sig
            for c, x, y, obs_eid in corrs:
                if world.budget_left < N_INT * 2 + 1:
                    break
                hi = world.intervene(x, +2.0, [y], n=N_INT)
                lo = world.intervene(x, -2.0, [y], n=N_INT)
                a = [s[y] for s in hi.samples]
                b = [s[y] for s in lo.samples]
                d = sum(a) / len(a) - sum(b) / len(b)
                se = _se_diff(a, b)
                if abs(d) > EFFECT_Z * se:
                    sign = "+" if d > 0 else "-"
                    store.assert_claim(
                        f"causal({x},{y},{sign})", 0.85,
                        [hi.eid, lo.eid],
                        world.budget_total - world.budget_left)
        except BudgetExhausted:
            return
