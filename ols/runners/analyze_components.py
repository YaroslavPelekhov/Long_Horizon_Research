"""Paired component-ablation analysis (LMW v0.2, N=15)."""

from __future__ import annotations

import json
import statistics as st
import sys
from math import comb, sqrt
from pathlib import Path


def t_ci(deltas, alpha=0.05):
    n = len(deltas)
    m = st.fmean(deltas)
    sd = st.stdev(deltas) if n > 1 else 0.0
    se = sd / sqrt(n) if n > 0 else 0.0
    # two-sided t critical for df=n-1, alpha=0.05 -> 2.145 for n=15
    t_crit = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57,
              10: 2.23, 14: 2.145, 19: 2.093, 24: 2.064}.get(n - 1, 2.0)
    return m, sd, se, (m - t_crit * se, m + t_crit * se)


def sign_test_p(deltas, thr=0.05):
    wins = sum(1 for d in deltas if d > thr)
    losses = sum(1 for d in deltas if d < -thr)
    nontie = wins + losses
    if nontie == 0:
        return 1.0, wins, losses
    k = min(wins, losses)
    p1 = sum(comb(nontie, i) for i in range(k + 1)) / 2 ** nontie
    return min(1.0, 2 * p1), wins, losses


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    proj = Path(__file__).resolve().parent.parent.parent

    # Anchors (OLS-full, OLS-all-off) from prior 15s sweep
    anchors = json.load(open(proj / "lmw" / "ols_lmw_openai-gpt-4o-mini_15s_anchors.json"))
    # Components (4 new ablations) from this sweep
    comp = json.load(open(proj / "lmw" / "ols_lmw_openai-gpt-4o-mini_15s.json"))

    seeds = [r["master_seed"] for r in anchors["OLS-full"]]
    rps = {}
    for src in (anchors, comp):
        for tag, rows in src.items():
            if not isinstance(rows, list):
                continue
            if rows and isinstance(rows[0], dict) and "total_rps" in rows[0]:
                # paired by master_seed
                by_seed = {r["master_seed"]: r["total_rps"] for r in rows}
                rps[tag] = [by_seed[s] for s in seeds]

    print(f"=== LMW v0.2 component-ablation paired analysis (N={len(seeds)}) ===\n")
    print(f"{'condition':<22}  {'mean RPS':>10}  {'sd':>6}")
    print("-" * 45)
    for tag in ("OLS-full", "OLS-full-no-gate", "OLS-mem-off",
                "OLS-agn-off", "OLS-fut-off", "OLS-all-off"):
        if tag not in rps:
            continue
        m = st.fmean(rps[tag])
        sd = st.stdev(rps[tag])
        print(f"{tag:<22}  {m:>+10.3f}  {sd:>6.3f}")

    print()
    print("=== Paired Δ vs OLS-full (how much does ablating this component cost?) ===")
    print(f"{'ablation':<22}  {'paired Δ':>10}  {'95% CI':>22}  {'sign-p':>7}  {'w/l/t':>8}")
    print("-" * 78)
    anchor = rps["OLS-full"]
    for tag in ("OLS-full-no-gate", "OLS-mem-off", "OLS-agn-off",
                "OLS-fut-off", "OLS-all-off"):
        if tag not in rps:
            continue
        deltas = [a - b for a, b in zip(rps[tag], anchor)]
        m, sd, se, (lo, hi) = t_ci(deltas)
        p, w, l = sign_test_p(deltas)
        t = len(deltas) - w - l
        print(f"{tag:<22}  {m:>+10.3f}  [{lo:>+6.3f}, {hi:>+6.3f}]  {p:>7.3f}  {w:>2}/{l:>2}/{t:>2}")

    print()
    print("=== Paired Δ vs OLS-all-off (how much does adding this component buy?) ===")
    print(f"{'condition':<22}  {'paired Δ':>10}  {'95% CI':>22}  {'sign-p':>7}  {'w/l/t':>8}")
    print("-" * 78)
    anchor = rps["OLS-all-off"]
    for tag in ("OLS-full", "OLS-full-no-gate", "OLS-mem-off",
                "OLS-agn-off", "OLS-fut-off"):
        if tag not in rps:
            continue
        deltas = [a - b for a, b in zip(rps[tag], anchor)]
        m, sd, se, (lo, hi) = t_ci(deltas)
        p, w, l = sign_test_p(deltas)
        t = len(deltas) - w - l
        print(f"{tag:<22}  {m:>+10.3f}  [{lo:>+6.3f}, {hi:>+6.3f}]  {p:>7.3f}  {w:>2}/{l:>2}/{t:>2}")


if __name__ == "__main__":
    main()
