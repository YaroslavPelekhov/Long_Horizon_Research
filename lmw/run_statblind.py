"""Run Scripted-blind on Open-Ended LMW (5 master seeds, deterministic, $0)."""
from __future__ import annotations
import json, statistics as st
from openworld import consistency_selftest, run_curriculum
from statblind_agent import StatBlindAgent

SEEDS = [1, 2, 3, 4, 5]

def main():
    consistency_selftest()
    print("\n=== Scripted-blind (schema-blind statistical baseline) on Open-Ended LMW ===")
    dump = {"seeds": SEEDS}
    for nm, persist in (("StatBlind-Full", True), ("StatBlind-mem", False)):
        per = []
        for s in SEEDS:
            r = run_curriculum(StatBlindAgent, master_seed=s, persistent=persist)
            per.append((s, r["reached"], r["total_rps"]))
        ds = [p[1] for p in per]; ts = [p[2] for p in per]
        dump[nm] = per
        print(f"  {nm:<20} depth={st.fmean(ds):.2f}  total RPS={st.fmean(ts):+.3f}  "
              f"depths={ds}  totals={ts}")
    with open("statblind_5s.json", "w") as f:
        json.dump(dump, f, indent=1)
    print(" [written statblind_5s.json]")


if __name__ == "__main__":
    main()
