"""
Research Program Score (RPS) + axis-1/2/6 metrics.

RPS is NOT the final artifact's quality. It is the area under the curve of
*validated, retained, causally-grounded* knowledge vs budget, minus a penalty
for un-retracted false beliefs (integrity). A false claim that stays in the
theory SUBTRACTS.

A claim contributes to VRK at budget b iff:
  (i)  it was active at b,
  (ii) it is TRUE vs the SCM *at budget b* (truth can change after a regime
       shift -- this is the implicit re-test),
  (iii) its provenance is honest (cited experiments exist and are of a kind
        that could support the statement).
"""

from __future__ import annotations

from claims import ClaimStore
from oracle import Oracle
from world import World

FALSE_PENALTY = 1.5   # weight on un-retracted false beliefs
DEADEND_LAMBDA = 0.6  # (fix 2) RPS penalty weight on wasted dead-end budget
GRID = 60             # budget-grid resolution for the integral


# -- helpers -----------------------------------------------------------------
def _subdomain_of(exp, oracle: Oracle) -> str | None:
    for v in list(exp.interventions) + list(exp.measured):
        sd = oracle.subdomain_of_var(v)
        if sd is not None:
            return sd
    return None


def _provenance_honest(statement: str, prov: list[int], world: World) -> bool:
    by_id = {e.eid: e for e in world.log}
    if not prov or any(p not in by_id for p in prov):
        return False
    cited = [by_id[p] for p in prov]
    # Causal / no-effect / confounding claims REQUIRE interventional evidence.
    needs_intervention = (statement.startswith("causal(")
                          or statement.startswith("no_effect(")
                          or statement.startswith("confounded("))
    if needs_intervention and not any(e.kind == "intervene" for e in cited):
        return False
    return True


# -- the score ---------------------------------------------------------------
def score(world: World, store: ClaimStore, oracle: Oracle,
          regime_shift_at: float | None = None) -> dict:
    B = world.budget_total
    grid = [B * i / GRID for i in range(GRID + 1)]

    def vrk(b: float) -> float:
        v = 0.0
        for c in store.active_at(b):
            if not oracle.is_true(c.statement, b):
                continue
            if not _provenance_honest(c.statement, c.provenance, world):
                continue
            v += oracle.fact_value(c.statement)
        return v

    def vrk_star(b: float) -> float:
        return oracle.optimal_value(b)

    # trapezoidal integrals over the budget grid
    def integ(fn) -> float:
        tot = 0.0
        for i in range(len(grid) - 1):
            tot += 0.5 * (fn(grid[i]) + fn(grid[i + 1])) * (grid[i + 1] - grid[i])
        return tot

    area = integ(vrk)
    area_star = integ(vrk_star) or 1e-9
    rps_norm = area / area_star

    # integrity penalty: false claims that sit active, weighted by how long
    integrity = 0.0
    for c in store.claims:
        end = c.retracted_at if c.retracted_at is not None else B
        # sample truth at the mid-life of the claim
        mid = min(B, (c.budget_stamp + end) / 2.0)
        if not oracle.is_true(c.statement, mid):
            dur = max(0.0, end - c.budget_stamp)
            integrity += FALSE_PENALTY * (dur / B) * (1.0 + oracle.fact_value(c.statement))

    # -- axis 6: dead-end regret (computed early; also feeds RPS) --------
    fake_sds = set(oracle.fake_subdomains())
    de_point = oracle.deadend_point()
    regret = 0.0
    for e in world.log:
        if _subdomain_of(e, oracle) in fake_sds and e.budget_at >= de_point:
            regret += (0.5 if e.kind == "observe" else 1.0) * e.n
    axis6_deadend_regret = regret / B

    # (fix 2) RPS now penalizes wasted dead-end budget (opportunity cost),
    # so failing to abandon hurts the score itself, not just a side metric.
    rps = (rps_norm
           - integrity / max(1.0, oracle.max_value())
           - DEADEND_LAMBDA * axis6_deadend_regret)

    # -- axis 1: agenda value ratio -------------------------------------
    final_true_value = sum(
        oracle.fact_value(c.statement)
        for c in store.active()
        if oracle.is_true(c.statement, B)
        and _provenance_honest(c.statement, c.provenance, world)
    )
    axis1 = final_true_value / max(1e-9, oracle.optimal_value(B))

    # -- axis 2: retention + revision -----------------------------------
    half = B / 2.0
    early_true = [c for c in store.claims
                  if c.budget_stamp <= half and oracle.is_true(c.statement, c.budget_stamp)]
    retained = [c for c in early_true
                if c.status == "active" and oracle.is_true(c.statement, B)]
    axis2_retention = len(retained) / max(1, len(early_true))

    # (fix 3) axis2_revision: of the claims the agent ASSERTED before the
    # shift that the shift turned false, what fraction did it retract?
    # No retraction when truth changed = 0.0 (failure), NOT a misleading 1.0.
    # None only when no regime shift / agent never asserted an affected claim.
    axis2_revision = None
    if regime_shift_at is not None:
        affected = [c for c in store.claims
                    if c.budget_stamp < regime_shift_at
                    and oracle.is_true(c.statement, regime_shift_at - 1e-6)
                    and not oracle.is_true(c.statement, B)]
        if affected:
            revised = [c for c in affected
                       if c.status == "retracted"
                       and c.retracted_at is not None
                       and c.retracted_at >= regime_shift_at]
            axis2_revision = len(revised) / len(affected)

    return {
        "RPS": round(rps, 4),
        "RPS_norm": round(rps_norm, 4),
        "integrity_penalty": round(integrity / max(1.0, oracle.max_value()), 4),
        "axis1_agenda_ratio": round(axis1, 4),
        "axis2_retention": round(axis2_retention, 4),
        "axis2_revision": (round(axis2_revision, 4) if axis2_revision is not None else None),
        "axis6_deadend_regret": round(axis6_deadend_regret, 4),
        "final_true_value": round(final_true_value, 2),
        "oracle_max": round(oracle.optimal_value(B), 2),
    }
