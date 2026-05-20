"""
Open-Ended LMW v0 -- counterfactual harness over the competence-gated
curriculum. Reports stages-reached and total RPS per variant, averaged over
master seeds. Consistency self-test MUST pass first (abort otherwise).
No API, deterministic.
"""

from __future__ import annotations

import statistics as st

from agents import ConnectorAgent, RandomAgent
from openworld import consistency_selftest, run_curriculum

SEEDS = [1, 2, 3, 4, 5]
# (factory, persistent-theory-across-stages?)
VARIANTS = {
    "Full":     (lambda: ConnectorAgent(True, True, True),  True),
    "-mem":     (lambda: ConnectorAgent(False, True, True),  False),  # theory wiped at each horizon
    "-goal":    (lambda: ConnectorAgent(True, False, True),  True),
    "-abandon": (lambda: ConnectorAgent(True, True, False),  True),
    "naive":    (RandomAgent,                                 True),
}


def main():
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    consistency_selftest()                       # L2 insurance -- abort if fails

    print(f"\n=== Open-Ended LMW v0: competence-gated curriculum "
          f"(avg over {len(SEEDS)} master seeds, tau=0.5) ===")
    print(f"{'variant':<10}{'avg_stages_reached':>20}{'avg_total_RPS':>16}"
          f"{'depth_dist':>16}")
    print("-" * 62)
    res = {}
    for nm, (fac, persist) in VARIANTS.items():
        reached, total = [], []
        for s in SEEDS:
            r = run_curriculum(fac, master_seed=s, persistent=persist)
            reached.append(r["reached"]); total.append(r["total_rps"])
        res[nm] = (st.fmean(reached), st.fmean(total))
        print(f"{nm:<10}{st.fmean(reached):>20.2f}{st.fmean(total):>16.3f}"
              f"{str(sorted(reached)):>16}")
    f = res["Full"]
    print(f"\n depth: Full {f[0]:.2f} vs -mem {res['-mem'][0]:.2f} "
          f"vs -goal {res['-goal'][0]:.2f} vs -abandon {res['-abandon'][0]:.2f} "
          f"vs naive {res['naive'][0]:.2f}")
    deeper = f[0] >= max(res[v][0] for v in ("-goal", "-abandon", "naive"))
    print(f" sanity: Full reaches >= goal/abandon/naive depth: "
          f"{'YES' if deeper else 'NO'}")
    print(" NOTE (honest): the scripted agent re-derives each stage from "
          "scratch (stage vars are namespaced) -> cross-stage axis-2 transfer "
          "is NOT exercised by this dummy; Full vs -mem on DEPTH may tie. The "
          "open-ended rung's axis-2-across-stages needs a LEARNING agent "
          "(LLM that abstracts strategy). v0 validates: consistency invariant, "
          "competence-gated depth, and that weaker process (-goal/-abandon/"
          "naive) reaches shallower.")


if __name__ == "__main__":
    main()
