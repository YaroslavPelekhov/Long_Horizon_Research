"""
AutoDiscovery-algorithm adapter on Open-Ended LMW.

Mirrors run_open_llm.py: per-seed runs through the curriculum with persistent
ClaimStore (Full) vs wiped between stages (-mem, the mandatory axis-2
ablation per BENCHMARK_PROTOCOL_v0.md §5). Per-seed JSON dump.

Run:
  set -a; . ../autodiscovery/.env.local; set +a
  ../autodiscovery/.venv/Scripts/python.exe run_autodisc.py
"""

from __future__ import annotations

import json
import os
import statistics as st

from autodisc_agent import AutoDiscoveryAgent
from openworld import consistency_selftest, run_curriculum


SEEDS = list(range(1, 1 + int(os.getenv("AUTODISC_SEEDS", "3"))))
MAX_ITER = int(os.getenv("AUTODISC_MAX_ITER", "12"))
N_BEL = int(os.getenv("AUTODISC_N_BEL", "3"))
LLM_CAP = int(os.getenv("AUTODISC_LLM_CAP", "40"))
MODEL = os.getenv("AUTODISC_MODEL", "openai/gpt-4o-mini")


def _factory():
    return AutoDiscoveryAgent(
        model=MODEL,
        max_iterations=MAX_ITER,
        n_belief_samples=N_BEL,
        llm_call_cap=LLM_CAP,
    )


def main():
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    consistency_selftest()
    print(f"\n=== AutoDiscovery-algorithm adapter on Open-Ended LMW ===")
    print(f" model={MODEL} max_iter={MAX_ITER} n_bel={N_BEL} cap={LLM_CAP}")
    print(f" seeds={SEEDS}")
    print(f"{'variant':<14}{'avg_depth':>12}{'avg_total_RPS':>16}"
          f"{'depths':>16}")
    print("-" * 60)
    dump = {"model": MODEL, "max_iter": MAX_ITER, "n_bel": N_BEL,
            "cap": LLM_CAP, "seeds": SEEDS}
    res = {}
    for nm, persist in (("AutoDisc-Full", True), ("AutoDisc-mem", False)):
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
    out = f"autodisc_{MODEL.split('/')[-1]}_{len(SEEDS)}s.json"
    with open(out, "w") as f:
        json.dump(dump, f, indent=1)
    print(f" [written {out}]")
    f0, m0 = res["AutoDisc-Full"], res["AutoDisc-mem"]
    print(f"\n AutoDiscovery-algorithm leaderboard entry "
          f"(category B/LLM, tier T1, adapter — not original codebase):")
    print(f"   AutoDisc-Full: depth {f0[0]:.2f}, total RPS {f0[1]:.3f}")
    print(f"   AutoDisc-mem : depth {m0[0]:.2f}, total RPS {m0[1]:.3f}")
    print(f"   axis-2 (carried ClaimStore) value: "
          f"depth {f0[0]-m0[0]:+.2f}, total RPS {f0[1]-m0[1]:+.3f}")


if __name__ == "__main__":
    main()
