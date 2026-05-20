"""Detailed paired analysis of the DB sweep dump (ols_db_*_20t.json)."""

from __future__ import annotations

import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    proj = Path(__file__).resolve().parent.parent.parent
    p = proj / "lmw" / "ols_db_openai-gpt-4o-mini_20t.json"
    d = json.load(open(p))
    full = d["per_task"]["OLS-full"]
    off = d["per_task"]["OLS-all-off"]
    assert len(full) == len(off)
    by_full = {r["task_id"]: r for r in full}
    by_off = {r["task_id"]: r for r in off}
    keys = list(by_full.keys())

    print("=== Paired DB results (per task) ===")
    hdr_t = "task_id"
    print(f"{hdr_t[:50]:<50}  {'full':>6}  {'off':>6}  {'delta':>7}")
    print("-" * 80)
    deltas = []
    for k in keys:
        f = by_full[k]["HMS"]
        o = by_off[k]["HMS"]
        dlt = f - o
        deltas.append(dlt)
        print(f"{k[-50:]:<50}  {f:>6.2f}  {o:>6.2f}  {dlt:>+7.2f}")

    print()
    print(f"mean delta: {st.fmean(deltas):+.4f}")
    print(f"paired SD : {st.stdev(deltas):.3f}")
    se = st.stdev(deltas) / (len(deltas) ** 0.5)
    print(f"paired SE : {se:.3f}")
    print(f"95% CI    : [{st.fmean(deltas) - 1.96 * se:+.3f}, "
          f"{st.fmean(deltas) + 1.96 * se:+.3f}]")
    wins = sum(1 for d in deltas if d > 0.05)
    losses = sum(1 for d in deltas if d < -0.05)
    ties = len(deltas) - wins - losses
    print(f"wins/losses/ties (|Δ|>0.05): {wins}/{losses}/{ties}")

    # sign test p-value (two-sided)
    # under H0 of P(win)=P(loss)=0.5, count non-tie cases as Bernoulli(0.5)
    n_nontie = wins + losses
    if n_nontie > 0:
        from math import comb
        # P(at least as extreme as observed) — two-sided
        k_obs = min(wins, losses)
        p_one = sum(comb(n_nontie, i) for i in range(0, k_obs + 1)) / 2**n_nontie
        p_two = min(1.0, 2 * p_one)
        print(f"sign-test p (two-sided): {p_two:.3f} (n_nontie={n_nontie})")

    print()
    print("=== By domain ===")
    dom = defaultdict(list)
    for k in keys:
        dom[by_full[k]["domain"]].append((by_full[k]["HMS"], by_off[k]["HMS"]))
    for d_, pairs in sorted(dom.items()):
        fs = [p[0] for p in pairs]
        os = [p[1] for p in pairs]
        dlts = [p[0] - p[1] for p in pairs]
        print(f"  {d_:<12}: N={len(pairs):>2}  "
              f"full_HMS={st.fmean(fs):.2f}  off_HMS={st.fmean(os):.2f}  "
              f"mean_delta={st.fmean(dlts):+.3f}")

    print()
    print("=== Task-failure modes (HMS=0 episodes) ===")
    for tag, rows in (("OLS-full", full), ("OLS-all-off", off)):
        no_sub = sum(1 for r in rows if not r.get("submitted"))
        zero_hms = sum(1 for r in rows if r["HMS"] == 0.0)
        print(f"  {tag:<14}: empty_submission={no_sub}/{len(rows)}  "
              f"HMS=0: {zero_hms}/{len(rows)}")

    print()
    print("=== Resource use (mean per episode) ===")
    for tag, rows in (("OLS-full", full), ("OLS-all-off", off)):
        n_act = st.fmean(r["n_actions"] for r in rows)
        n_cl = st.fmean(r["n_claims_active"] for r in rows)
        ax1 = st.fmean(r.get("axis1_ratio", 0.0) for r in rows)
        ax6 = st.fmean(r.get("axis6_regret", 0.0) for r in rows)
        wt = st.fmean(r.get("wall_time_s", 0.0) for r in rows)
        print(f"  {tag:<14}: n_actions={n_act:.1f}  n_claims={n_cl:.1f}  "
              f"axis1={ax1:.2f}  axis6={ax6:.2f}  wall={wt:.0f}s")

    print()
    print("=== Failure-mode hot tasks (HMS=0 in BOTH conditions) ===")
    both_zero = [k for k in keys
                 if by_full[k]["HMS"] == 0.0 and by_off[k]["HMS"] == 0.0]
    print(f"  {len(both_zero)}/{len(keys)} tasks both zero (= task-defeats-LLM, "
          f"not overlay-attributable)")
    for k in both_zero:
        print(f"    {k}")


if __name__ == "__main__":
    main()
