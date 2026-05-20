"""
Step B benchmark: does outer-loop scaffolding move RPS, and is the metric
sensitive to a REAL agent's memory?

Compares on the construct-valid harness (dev schemas), same structures/seeds:
  Full           scripted competent reference (upper-ish bound)
  -mem(dummy)     scripted, memory ablated
  LLM-strong      LLM + structured theory/agenda/abandon scaffold
  LLM-strong-mem  same LLM, persistent-theory scaffold ablated
  LLM-base        thin LLM, no scaffold

Cheap: gpt-4o-mini, 2 schemas x 2 struct seeds, 1 noise seed, call cap 36.
Run:
  set -a; . ../autodiscovery/.env.local; set +a
  ../autodiscovery/.venv/Scripts/python.exe run_llm.py
"""

from __future__ import annotations

from agents import ConnectorAgent
from claims import ClaimStore
from harness import _budget_for
from llm_agent import BaselineLLM, StrongLLM
from oracle import Oracle
from scm import Schema, make_structure
from scorer import score
from world import World

COLS = ["RPS", "axis1_agenda_ratio", "axis2_retention",
        "axis6_deadend_regret", "final_true_value"]


def run(agent, structure, ns):
    w = World(structure=structure, noise_seed=ns, budget=_budget_for(structure))
    s = ClaimStore()
    agent.run(w, s)
    return score(w, s, Oracle(w._scm))


def main() -> None:
    schemas = [Schema("d2", n_clusters=3, chain_len=1, n_latents=2, real_decoys=1),
               Schema("d3", n_clusters=3, chain_len=2, n_latents=3, real_decoys=1)]
    seeds = [11, 22]
    agg: dict[str, list] = {k: [] for k in
                            ["Full", "-mem(dummy)", "LLM-strong",
                             "LLM-strong-mem", "LLM-base"]}
    total_calls = 0
    for sc in schemas:
        for ss in seeds:
            st = make_structure(ss, sc)
            agg["Full"].append(run(ConnectorAgent(True, True, True), st, 1))
            agg["-mem(dummy)"].append(run(ConnectorAgent(False, True, True), st, 1))
            a1 = StrongLLM(memory=True, max_calls=36)
            agg["LLM-strong"].append(run(a1, st, 1))
            a2 = StrongLLM(memory=False, max_calls=36)
            agg["LLM-strong-mem"].append(run(a2, st, 1))
            a3 = BaselineLLM(max_calls=36)
            agg["LLM-base"].append(run(a3, st, 1))
            total_calls += a1.calls + a2.calls + a3.calls
            print(f"  [{sc.name} s{ss}] calls: strong {a1.calls} "
                  f"strong-mem {a2.calls} base {a3.calls}")

    head = f"{'agent':<16}" + "".join(f"{c:>22}" for c in COLS)
    print("\n" + head); print("-" * len(head))
    for name, ms in agg.items():
        avg = {c: round(sum(m[c] for m in ms) / len(ms), 4) for c in COLS}
        print(f"{name:<16}" + "".join(f"{avg[c]:>22}" for c in COLS))
    print(f"\ntotal LLM calls: {total_calls} (gpt-4o-mini)")
    print("Read: does LLM-strong > LLM-base (scaffold helps?) and "
          "LLM-strong > LLM-strong-mem (metric sensitive to a real agent's "
          "memory?).")


if __name__ == "__main__":
    main()
