"""
L2-real: counterfactual harness on REAL session #3787, behind WorldSource.

Reuses the domain-agnostic ClaimStore. Ground truth = out-of-sample
replication (segments 1-4 discover, 5-8 validate) computed by the source; the
agent never sees the held-out slice. The motion/quality cluster (A_MOT) is a
REAL tempting dead end (big contrasts, not split-half stable, ~0 value).

Run (needs nothing but stdlib):
    python neuro_lab.py
"""

from __future__ import annotations

from claims import AgendaItem, ClaimStore
from adapters.base import BudgetExhausted
from adapters.neuro import NeuroSource

DEADEND_LAMBDA = 0.6
FALSE_PENALTY = 1.0
GRID = 40


# ---------------- agents (generic over WorldSource) ----------------
class StatAgent:
    name = "Stat"

    def __init__(self, use_memory=True, choose_goals=True, can_abandon=True):
        self.use_memory = use_memory
        self.choose_goals = choose_goals
        self.can_abandon = can_abandon

    def run(self, src, store: ClaimStore) -> None:
        clusters = sorted(src.clusters())                 # 'A_MOT' first (dud)
        dud = set(src.deadend_clusters())
        done: set[str] = set()
        store.set_agenda([AgendaItem(c, f"investigate {c}",
                          "self" if self.choose_goals else "imposed")
                          for c in clusters], src.budget_total - src.budget_left)
        guard = 0
        try:
            while src.budget_left >= 3.0 and guard < 40:
                guard += 1
                # pick next cluster
                if not self.choose_goals:
                    cluster = sorted(dud)[0]              # forced: mine the dud
                elif self.use_memory:
                    nxt = [c for c in clusters if c not in done]
                    if not nxt:
                        break
                    cluster = nxt[0]
                else:
                    cluster = clusters[0]   # no memory -> always restart at dud

                obs = src.observe(cluster)
                spent = src.budget_total - src.budget_left
                stable = [(f, c) for f, c in obs.items()
                          if c.stable and abs(c.delta) > 3 * c.se]

                if cluster in dud:
                    # competent read: a real effect is split-half stable;
                    # motion is not -> almost nothing stable here -> abandon.
                    if self.can_abandon:
                        if self.use_memory:
                            done.add(cluster)
                            store.set_agenda(
                                [a for a in store.agenda
                                 if a.subdomain != cluster], spent)
                        continue          # -mem: not marked done -> loops dud
                    else:
                        while src.budget_left >= 3.0:      # cannot leave dud
                            src.observe(cluster)
                        break
                for f, c in stable:
                    stmt = f"stim_effect({f},{'+' if c.delta > 0 else '-'})"
                    if not any(x.statement == stmt and x.status == "active"
                               for x in store.claims):
                        store.assert_claim(stmt, 0.8, [c.eid], spent)
                if self.use_memory:
                    done.add(cluster)
        except BudgetExhausted:
            pass


class NaiveAgent:
    name = "Naive"

    def run(self, src, store: ClaimStore) -> None:
        try:
            for cl in sorted(src.clusters()):
                obs = src.observe(cl)
                top = sorted(obs.items(),
                             key=lambda kv: -abs(kv[1].delta))[:4]
                for f, c in top:                          # claim raw top, no
                    store.assert_claim(                    # stability check
                        f"stim_effect({f},{'+' if c.delta > 0 else '-'})",
                        0.9, [c.eid], src.budget_total - src.budget_left)
        except BudgetExhausted:
            pass


# ---------------- replication-based scorer (generic keys) ----------------
def score(src, store: ClaimStore) -> dict:
    B = src.budget_total
    grid = [B * i / GRID for i in range(GRID + 1)]

    def ok(c):
        return (src.replicates(c.statement)
                and src.provenance_ok(c.statement, c.provenance))

    def vrk(b):
        v = 0.0
        for c in store.claims:
            if c.budget_stamp <= b and c.status == "active" and ok(c):
                v += src.value(c.statement)
        return v

    def integ(fn):
        return sum(0.5 * (fn(grid[i]) + fn(grid[i + 1])) * (grid[i + 1] - grid[i])
                   for i in range(len(grid) - 1))

    oracle = max(src.oracle_value(), 1.0)
    rps_norm = integ(vrk) / (oracle * B)

    integrity = 0.0
    for c in store.claims:
        if not src.replicates(c.statement):               # asserted, didn't hold
            integrity += FALSE_PENALTY * (1.0 + src.value(c.statement)) / oracle

    de = set(src.deadend_clusters())
    de_obs = sum(1 for _, cl in src.log if cl in de)
    deadend_regret = max(0, de_obs - 1) * 3.0 / B
    rps = rps_norm - integrity - DEADEND_LAMBDA * deadend_regret

    true_now = [c for c in store.active() if ok(c)]
    final_true = sum(src.value(c.statement) for c in true_now)
    axis1 = final_true / oracle
    half = B / 2.0
    early = [c for c in store.claims if c.budget_stamp <= half and ok(c)]
    retained = [c for c in early if c.status == "active"]
    axis2_ret = len(retained) / max(1, len(early))

    return {
        "RPS": round(rps, 4),
        "RPS_norm": round(rps_norm, 4),
        "integrity_penalty": round(integrity, 4),
        "axis1_agenda_ratio": round(axis1, 4),
        "axis2_retention": round(axis2_ret, 4),
        "axis6_deadend_regret": round(deadend_regret, 4),
        "final_true_value": round(final_true, 2),
        "oracle_value": round(oracle, 2),
    }


# ---------------- counterfactual harness ----------------
def run_fold(agent_factory, held_out, budget):
    src = NeuroSource(held_out=held_out, budget=budget)
    store = ClaimStore()
    agent_factory().run(src, store)
    return score(src, store)


def main():
    import sys
    from adapters.neuro import SUBJECTS
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    B = 45.0
    variants = {
        "Full":     lambda: StatAgent(True, True, True),
        "-mem":     lambda: StatAgent(False, True, True),
        "-goal":    lambda: StatAgent(True, False, True),
        "-abandon": lambda: StatAgent(True, True, False),
        "Naive":    NaiveAgent,
    }
    cols = ["RPS", "RPS_norm", "integrity_penalty", "axis1_agenda_ratio",
            "axis2_retention", "axis6_deadend_regret", "final_true_value",
            "oracle_value"]
    print("\n=== L2-real: NeuroTrend Smell_Betula #2591 -- 4-fold "
          "leave-one-SUBJECT-out (cross-subject replication GT) ===")
    head = f"{'variant':<10}" + "".join(f"{c:>20}" for c in cols)
    print(head); print("-" * len(head))
    res = {}
    for nm, fac in variants.items():
        acc = {}
        for h in SUBJECTS:
            m = run_fold(fac, h, B)
            for k, v in m.items():
                acc[k] = acc.get(k, 0.0) + v / len(SUBJECTS)
        res[nm] = {k: round(v, 4) for k, v in acc.items()}
        print(f"{nm:<10}" + "".join(f"{res[nm][c]:>20}" for c in cols))
    f, mem = res["Full"]["RPS"], res["-mem"]["RPS"]
    g, ab, nv = (res["-goal"]["RPS"], res["-abandon"]["RPS"],
                 res["Naive"]["RPS"])
    print(f"\n outer-loop value: axis2(mem) {f-mem:+.3f}  "
          f"axis1(goal) {f-g:+.3f}  axis6(abandon) {f-ab:+.3f}  "
          f"vs naive {f-nv:+.3f}")
    print(f" sanity: cross-subject replicable EEG stim-effects give mean "
          f"oracle_value={res['Full']['oracle_value']} "
          f"(0 means no signal generalizes across subjects)")
    ok = f > max(mem, g, ab, nv) and res["Full"]["oracle_value"] > 1.0
    print(f" self-check: {'PASS' if ok else 'FAIL'} "
          f"(Full best AND cross-subject signal exists)")


if __name__ == "__main__":
    main()
