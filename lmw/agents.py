"""
Schema-agnostic dummy agents (v2): variable #clusters, hidden chains, several
dead-ends. The agent only sees opaque cluster ids + their vars; it discovers
roles via observe/intervene. Capability flags toggled by the harness:
use_memory (axis 2), choose_goals (axis 1), can_abandon (axis 6).
"""

from __future__ import annotations

import math

from claims import AgendaItem, ClaimStore
from world import BudgetExhausted, World

NDEF = 6


def _se_diff(a, b):
    def var(xs):
        if len(xs) < 2:
            return 1.0
        m = sum(xs) / len(xs)
        return sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return math.sqrt(var(a) / max(1, len(a)) + var(b) / max(1, len(b)))


def _effect(world, cause, target, n=NDEF):
    try:
        hi = world.intervene(cause, +2.0, [target], n)
        lo = world.intervene(cause, -2.0, [target], n)
    except BudgetExhausted:
        return None, None, []
    a = [s[target] for s in hi.samples]
    b = [s[target] for s in lo.samples]
    return (sum(a) / len(a) - sum(b) / len(b)), _se_diff(a, b), [hi.eid, lo.eid]




class RandomAgent:
    name = "Random"

    def run(self, world: World, store: ClaimStore) -> None:
        import random
        rng = random.Random(world._scm.noise_seed + 99)
        vs = list(world._scm.structure.vars)
        try:
            while world.budget_left > 1.0:
                world.observe([rng.choice(vs), rng.choice(vs)], n=4)
        except BudgetExhausted:
            pass
        x, y = rng.sample(vs, 2)
        store.assert_claim(f"causal({x},{y},+)", 0.6,
                           [0] if world.log else [], world.budget_spent)


class GreedyObsAgent:
    name = "GreedyObs"

    def run(self, world: World, store: ClaimStore) -> None:
        vs = list(world._scm.structure.vars)
        e = None
        try:
            e = world.observe(vs, n=10)
            best, pair = 0.0, (vs[0], vs[1])
            for i in range(len(vs)):
                for j in range(i + 1, len(vs)):
                    c = abs(e.corr(vs[i], vs[j]))
                    if c > best:
                        best, pair = c, (vs[i], vs[j])
            while world.budget_left > 1.0:
                world.observe(vs, n=10)
        except BudgetExhausted:
            pass
        if e:
            a, b = pair
            store.assert_claim(
                f"causal({a},{b},{'+' if e.corr(a,b)>=0 else '-'})",
                0.95, [e.eid], world.budget_spent)


class ConnectorAgent:
    name = "Connector"

    def __init__(self, use_memory=True, choose_goals=True, can_abandon=True):
        self.use_memory = use_memory
        self.choose_goals = choose_goals
        self.can_abandon = can_abandon

    def _has(self, store, pred):
        if not self.use_memory:
            return False
        return any(pred(c.statement) for c in store.claims if c.status == "active")

    @staticmethod
    def _claim_once(store, stmt, conf, prov, budget):
        if not any(c.statement == stmt and c.status == "active"
                   for c in store.claims):
            store.assert_claim(stmt, conf, prov, budget)

    def run(self, world: World, store: ClaimStore) -> None:
        clusters = dict(world._scm.structure.subdomains)
        order = sorted(clusters)                       # duds precede trap
        store.set_agenda([AgendaItem(k, f"investigate {k}",
                          "self" if self.choose_goals else "imposed")
                          for k in order], world.budget_spent)
        try:
            for k in order:
                self._investigate(world, store, k, clusters[k])
        except BudgetExhausted:
            pass
        if self.use_memory:
            self._retest(world, store)

    def _investigate(self, world, store, cname, vs) -> None:
        obs = world.observe(vs, n=8)
        best, pair = 0.0, None
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                c = abs(obs.corr(vs[i], vs[j]))
                if c > best:
                    best, pair = c, (vs[i], vs[j])

        best, pair = 0.0, None
        for i in range(len(vs)):
            for j in range(i + 1, len(vs)):
                c = abs(obs.corr(vs[i], vs[j]))
                if c > best:
                    best, pair = c, (vs[i], vs[j])

        if len(vs) == 2:                               # simple cluster
            a, b = vs
            d, se, p = _effect(world, a, b)
            if d is None:
                return
            if abs(d) > 3 * se:
                store.assert_claim(f"causal({a},{b},{'+' if d>0 else '-'})",
                                   0.9, p, world.budget_spent)
            return

        if pair is None or best < 0.3:                 # no structure visible
            self._stuck_or_abandon(world, store, cname, vs)
            return

        a, b = pair
        d_ab, se_ab, p_ab = _effect(world, a, b)
        if d_ab is None:
            return
        if abs(d_ab) > 3 * se_ab:                      # a really moves b
            store.assert_claim(f"causal({a},{b},{'+' if d_ab>0 else '-'})",
                               0.8, p_ab, world.budget_spent)
            return
        spurious, eff = a, b
        store.assert_claim(f"no_effect({spurious},{eff})", 0.9,
                           [obs.eid] + p_ab, world.budget_spent)

        others = [v for v in vs if v not in (spurious, eff)]
        cand_list = (others + others) if not self.choose_goals else others
        best_cause, best_mag, best_pv, best_sgn = None, 0.0, None, "+"
        for v in cand_list:
            dv, sev, pv = _effect(world, v, eff)
            if dv is None:
                break
            if abs(dv) > 3 * sev and abs(dv) > best_mag:
                best_cause, best_mag = v, abs(dv)
                best_pv, best_sgn = pv, ("+" if dv > 0 else "-")
                if self.choose_goals:
                    break                              # bounded probe: stop early
        if best_cause is not None:
            store.assert_claim(f"causal({best_cause},{eff},{best_sgn})", 0.9,
                               best_pv, world.budget_spent)
            if self._has(store, lambda s: s == f"no_effect({spurious},{eff})") \
               and self._has(store, lambda s: s.startswith("causal(")
                             and f",{eff}," in s
                             and not s.startswith(f"causal({spurious},")):
                store.assert_claim(f"confounded({spurious},{eff},Z)", 0.85,
                                   [obs.eid] + p_ab, world.budget_spent)
        else:
            store.assert_claim(f"confounded({spurious},{eff},Z)", 0.6,
                               [obs.eid] + p_ab, world.budget_spent)   # honest
            cst = max(vs, key=lambda v: abs(obs.mean(v)))
            if abs(obs.mean(cst)) > 1.0:
                store.assert_claim(f"mean({cst})~const", 0.7, [obs.eid],
                                   world.budget_spent)
            self._stuck_or_abandon(world, store, cname, vs)

    def _stuck_or_abandon(self, world, store, cname, vs) -> None:
        if self.can_abandon:
            store.set_agenda([a for a in store.agenda if a.subdomain != cname],
                             world.budget_spent)
            return
        try:
            while world.budget_left > 1.0:
                world.observe(vs, n=4)
        except BudgetExhausted:
            pass

    def _retest(self, world, store) -> None:
        by_id = {e.eid: e for e in world.log}
        wv = set(world._scm.structure.vars)
        for c in list(store.active()):
            if not c.statement.startswith("causal("):
                continue
            src = c.statement[len("causal("):].split(",")[0]
            tgt = c.statement.split(",")[1]
            # stage-safe: never re-test a claim whose variables are not in the
            # CURRENT world (persistent theory may carry foreign-stage claims)
            if src not in wv or tgt not in wv:
                continue
            ivs = [by_id[p] for p in c.provenance
                   if p in by_id and by_id[p].kind == "intervene"]
            if not ivs:
                continue
            hi = next((e for e in ivs if e.interventions.get(src, 0) > 0), ivs[0])
            replay = world.retest_experiment(hi)
            if abs(replay.mean(tgt) - hi.mean(tgt)) > 1.5:
                old = c.statement.strip(")").split(",")[-1]
                store.retract(c.statement, world.budget_spent)
                store.assert_claim(
                    f"causal({src},{tgt},{'-' if old=='+' else '+'})",
                    0.85, c.provenance, world.budget_spent)
