"""
Counterfactual ablation harness over a SCHEMA FAMILY (v2).

Budget is derived per-structure from oracle.reference_budget()*TIGHTNESS so the
tight-budget pressure is constant across schemas of different shapes. Construct
validity = ordering Full > {-mem,-goal,-abandon} > naive holds aggregated over a
dev family AND on a held-out schema whose shape never appeared in dev.
"""

from __future__ import annotations

from dataclasses import replace

from agents import ConnectorAgent, GreedyObsAgent, RandomAgent
from claims import ClaimStore
from oracle import Oracle
from scm import SCM, Schema, make_structure
from scorer import score
from world import World

VARIANTS = {
    "Full":      lambda: ConnectorAgent(True, True, True),
    "-mem":      lambda: ConnectorAgent(False, True, True),
    "-goal":     lambda: ConnectorAgent(True, False, True),
    "-abandon":  lambda: ConnectorAgent(True, True, False),
    "Random":    RandomAgent,
    "GreedyObs": GreedyObsAgent,
}
TIGHTNESS = 1.05


def _budget_for(structure) -> float:
    return round(Oracle(SCM(structure, 0)).reference_budget() * TIGHTNESS, 1)


def run_single(agent, structure, noise_seed: int) -> dict:
    budget = _budget_for(structure)
    w = World(structure=structure, noise_seed=noise_seed, budget=budget)
    s = ClaimStore()
    agent.run(w, s)
    rs = structure.schema.t_star if structure.schema.regime_shift else None
    return score(w, s, Oracle(w._scm), regime_shift_at=rs)


def table_over(schemas: list[Schema], struct_seeds: list[int],
                noise_seeds: list[int]) -> dict[str, dict]:
    acc: dict[str, dict[str, float]] = {}
    runs = 0
    for sc in schemas:
        for ss in struct_seeds:
            structure = make_structure(ss, sc)
            if sc.regime_shift:
                ob = _budget_for(structure)
                structure = make_structure(ss, replace(sc, t_star=round(0.55 * ob, 1)))
            for ns in noise_seeds:
                runs += 1
                for name, factory in VARIANTS.items():
                    m = run_single(factory(), structure, ns)
                    a = acc.setdefault(name, {})
                    for k, v in m.items():
                        if isinstance(v, (int, float)):
                            a[k] = a.get(k, 0.0) + v
    return {name: {k: round(v / runs, 4) for k, v in mv.items()}
            for name, mv in acc.items()}
