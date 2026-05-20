"""
Tree-Search-LLM-Judge (≈ AI Scientist v2 outer-loop kernel) adapter on
Open-Ended LMW. Mirrors run_autodisc.py: Full + −mem (axis-2 ablation per
BENCHMARK_PROTOCOL_v0.md §5), per-seed JSON dump.
"""

from __future__ import annotations

import json
import os
import statistics as st

from openworld import consistency_selftest, run_curriculum
from treesearch_agent import TreeSearchAgent


SEEDS = list(range(1, 1 + int(os.getenv("TS_SEEDS", "3"))))
MAX_ITER = int(os.getenv("TS_MAX_ITER", "12"))
THRESH = float(os.getenv("TS_THRESH", "0.7"))
LLM_CAP = int(os.getenv("TS_LLM_CAP", "40"))
MODEL = os.getenv("TS_MODEL", "openai/gpt-4o-mini")


def _factory():
    return TreeSearchAgent(model=MODEL, max_iterations=MAX_ITER,
                           assert_threshold=THRESH, llm_call_cap=LLM_CAP)


def main():
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    consistency_selftest()
    print(f"\n=== Tree-Search-LLM-Judge adapter on Open-Ended LMW ===")
    print(f" model={MODEL} max_iter={MAX_ITER} assert>={THRESH} cap={LLM_CAP}")
    print(f" seeds={SEEDS}")
    print(f"{'variant':<14}{'avg_depth':>12}{'avg_total_RPS':>16}"
          f"{'depths':>16}")
    print("-" * 60)
    dump = {"model": MODEL, "max_iter": MAX_ITER, "thresh": THRESH,
            "cap": LLM_CAP, "seeds": SEEDS}
    res = {}
    for nm, persist in (("TS-Full", True), ("TS-mem", False)):
        per = []
        for s in SEEDS:
            r = run_curriculum(_factory, master_seed=s, persistent=persist)
            per.append((s, r["reached"], r["total_rps"], r["per"]))
        ds = [p[1] for p in per]; ts = [p[2] for p in per]
        res[nm] = (st.fmean(ds), st.fmean(ts))
        dump[nm] = per
        print(f"{nm:<14}{st.fmean(ds):>12.2f}{st.fmean(ts):>16.3f}"
              f"{str(ds):>16}")
        print(f"   per-seed: {per}")
    out = f"treesearch_{MODEL.split('/')[-1]}_{len(SEEDS)}s.json"
    with open(out, "w") as f:
        json.dump(dump, f, indent=1)
    print(f" [written {out}]")
    f0, m0 = res["TS-Full"], res["TS-mem"]
    print(f"\n Tree-Search-LLM-Judge leaderboard entry "
          f"(category B/LLM, adapter — ≈ AI Scientist v2 outer-loop kernel, "
          f"NOT original codebase):")
    print(f"   TS-Full: depth {f0[0]:.2f}, total RPS {f0[1]:.3f}")
    print(f"   TS-mem : depth {m0[0]:.2f}, total RPS {m0[1]:.3f}")
    print(f"   axis-2 Δ: depth {f0[0]-m0[0]:+.2f}, "
          f"total RPS {f0[1]-m0[1]:+.3f}")


if __name__ == "__main__":
    main()
