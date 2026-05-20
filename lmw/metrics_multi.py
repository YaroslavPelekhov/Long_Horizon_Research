"""
Multi-metric leaderboard view (no new API calls; pure post-hoc analysis).

We already record per-seed RPS in the JSON dumps from run_open_llm.py /
run_autodisc.py / run_treesearch.py, and we know the costs from the
balance-tracking log. This module compiles:

  * RPS (mean, with 95% CI from per-seed where available)
  * Depth (mean)
  * Cost (\$ per evaluation, all seeds × conditions)
  * Mean composite rank across (RPS, Depth, Cost)
  * Pareto-front membership on (RPS↑, Cost↓)
  * Pairwise Spearman ρ across (RPS, Depth, Cost) -- if all three are
    highly correlated, RPS subsumes the others; if they diverge, the
    multi-metric view captures genuinely independent dimensions.

Honest scope: scripted entries have $0 cost and no per-seed RPS dump from
the current run_demo.py (could be added in v0.2). Calibration and
time-to-first-claim require ClaimStore-level dumps (slated for v0.2 of
the protocol).
"""

from __future__ import annotations

import json
import math
import os
import statistics as st
from pathlib import Path

try:
    from scipy import stats as _sps
    _HAVE_SCIPY = True
except ImportError:
    _HAVE_SCIPY = False


HERE = Path(__file__).parent


def _load(fn):
    p = HERE / fn
    if not p.exists():
        return None
    return json.load(open(p))


def _per_seed_rps(dump, key):
    """JSON dumps store dump[key] = [(seed, depth, total, ...), ...]."""
    return [row[2] for row in dump.get(key, [])]


def _per_seed_depth(dump, key):
    return [row[1] for row in dump.get(key, [])]


def _ci95(xs):
    """Half-width of a 95% t-CI on the mean. None if N<2."""
    n = len(xs)
    if n < 2:
        return None
    m, sd = st.fmean(xs), st.stdev(xs)
    if _HAVE_SCIPY:
        t = _sps.t.ppf(0.975, n - 1)
    else:
        t = 2.776 if n == 5 else (4.303 if n == 3 else 2.0)
    return t * sd / math.sqrt(n)


def _spearman(xs, ys):
    if _HAVE_SCIPY:
        return float(_sps.spearmanr(xs, ys).correlation)
    # manual fallback
    def ranks(a):
        s = sorted(range(len(a)), key=lambda i: a[i])
        r = [0] * len(a)
        for rk, idx in enumerate(s):
            r[idx] = rk
        return r
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1 - 6 * d2 / (n * (n * n - 1))


# ---- entries: (label, RPS_per_seed, depth_per_seed, cost_USD, tier) ----
# Per-seed numbers pulled from the JSON dumps in lmw/; costs from balance log.
ENTRIES = []


def _add_dump(label_pat, fn, cost, tier):
    """One JSON dump yields two rows: Full and -mem (the run-time labels vary)."""
    d = _load(fn)
    if not d:
        return
    # find the two condition keys
    cond_keys = [k for k in d.keys()
                 if isinstance(d[k], list) and d[k] and isinstance(d[k][0], list)]
    for ck in cond_keys:
        rps = _per_seed_rps(d, ck)
        dep = _per_seed_depth(d, ck)
        if not rps:
            continue
        ENTRIES.append((f"{label_pat}:{ck}", rps, dep, cost / max(1, len(cond_keys)),
                        tier))


# Generic LLM agent (run_open_llm.py)
_add_dump("Generic-LLM/gpt-4o-mini", "open_llm_gpt-4o-mini_5s.json",
          cost=0.005, tier="T0")
_add_dump("Generic-LLM/gpt-4o", "open_llm_gpt-4o_5s.json",
          cost=0.08, tier="T1")
_add_dump("Generic-LLM/llama-3.3-70b", "open_llm_llama-3.3-70b-instruct_3s.json",
          cost=0.008, tier="T0")
_add_dump("Generic-LLM/deepseek-chat", "open_llm_deepseek-chat_3s.json",
          cost=0.011, tier="T0")

# AutoDisc-algo adapter (run_autodisc.py)
_add_dump("AutoDisc/gpt-4o-mini", "autodisc_gpt-4o-mini_3s.json",
          cost=0.005, tier="T0")
_add_dump("AutoDisc/gpt-4o", "autodisc_gpt-4o_3s.json",
          cost=0.118, tier="T1")
_add_dump("AutoDisc/llama-3.3-70b", "autodisc_llama-3.3-70b-instruct_3s.json",
          cost=0.005, tier="T0")
_add_dump("AutoDisc/deepseek-chat", "autodisc_deepseek-chat_3s.json",
          cost=0.016, tier="T0")

# Tree-Search-LLM-Judge adapter
_add_dump("TreeSearch/gpt-4o-mini", "treesearch_gpt-4o-mini_3s.json",
          cost=0.002, tier="T0")

# Scripted ladder (no per-seed RPS dump yet -> single-element list = mean only)
for nm, rps_mean, dep_dist in [
    ("Scripted/Full", 0.459, [0, 0, 0, 2, 4]),
    ("Scripted/-mem", 0.360, [0, 0, 0, 2, 4]),
    ("Scripted/-goal", -0.003, [0, 0, 0, 0, 0]),
    ("Scripted/-abandon", -0.489, [0, 0, 0, 0, 0]),
    ("Scripted/naive", -0.200, [0, 0, 0, 0, 0]),
]:
    ENTRIES.append((nm, [rps_mean], dep_dist, 0.0, "T0"))


def main():
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    rows = []
    for (lbl, rps, dep, cost, tier) in ENTRIES:
        m_rps = st.fmean(rps); ci = _ci95(rps)
        m_dep = st.fmean(dep)
        rows.append(dict(label=lbl, tier=tier, rps=m_rps, ci=ci,
                         depth=m_dep, cost=cost, rps_seed=rps))

    # ranks (lower = better-ranked-first for cost; higher = better for rps,depth)
    by_rps = sorted(rows, key=lambda r: -r["rps"])
    by_dep = sorted(rows, key=lambda r: -r["depth"])
    by_cost = sorted(rows, key=lambda r: r["cost"])
    rk = {r["label"]: {} for r in rows}
    for i, r in enumerate(by_rps): rk[r["label"]]["rps"] = i + 1
    for i, r in enumerate(by_dep): rk[r["label"]]["depth"] = i + 1
    for i, r in enumerate(by_cost): rk[r["label"]]["cost"] = i + 1
    for r in rows:
        r["mean_rank"] = st.fmean(rk[r["label"]].values())

    # Pareto front on (RPS up, cost down)
    pareto = []
    for r in rows:
        dominated = False
        for o in rows:
            if o is r:
                continue
            if o["rps"] >= r["rps"] and o["cost"] <= r["cost"] \
               and (o["rps"] > r["rps"] or o["cost"] < r["cost"]):
                dominated = True
                break
        if not dominated:
            pareto.append(r["label"])

    # pairwise Spearman across (RPS, Depth, Cost) on the entries
    labels = [r["label"] for r in rows]
    rps_l = [r["rps"] for r in rows]
    dep_l = [r["depth"] for r in rows]
    cost_l = [r["cost"] for r in rows]
    sp_rd = _spearman(rps_l, dep_l)
    sp_rc = _spearman(rps_l, [-c for c in cost_l])  # ρ(RPS, -Cost) = "RPS vs cheapness"
    sp_dc = _spearman(dep_l, [-c for c in cost_l])

    # --- print multi-metric leaderboard ---
    print("\n=== Multi-metric leaderboard (Open-Ended LMW, no new API) ===")
    by_mean = sorted(rows, key=lambda r: r["mean_rank"])
    print(f"{'rank':>4}  {'system':<33} {'tier':<4} "
          f"{'RPS':>8} {'±CI':>7} {'depth':>6} {'cost$':>8} "
          f"{'mean_rk':>8} {'Pareto':>7}")
    print("-" * 92)
    for i, r in enumerate(by_mean):
        ci = f"±{r['ci']:.3f}" if r["ci"] is not None else "  —  "
        pf = "✓" if r["label"] in pareto else " "
        print(f"{i+1:>4}  {r['label']:<33} {r['tier']:<4} "
              f"{r['rps']:>8.3f} {ci:>7} {r['depth']:>6.2f} "
              f"{r['cost']:>8.4f} {r['mean_rank']:>8.2f} {pf:>7}")

    print("\nPareto front on (RPS↑, Cost↓):")
    for lab in pareto:
        r = next(x for x in rows if x["label"] == lab)
        print(f"  {lab:<33}  RPS={r['rps']:+.3f}  cost=${r['cost']:.4f}")

    print("\nPairwise Spearman ρ across metrics (N={}):".format(len(rows)))
    print(f"  ρ(RPS, Depth)       = {sp_rd:+.3f}")
    print(f"  ρ(RPS, −Cost)       = {sp_rc:+.3f}   (positive ⇒ RPS prefers cheaper)")
    print(f"  ρ(Depth, −Cost)     = {sp_dc:+.3f}")

    print("\nReading:")
    def _strength(r):
        a = abs(r)
        return ("strong" if a > 0.8 else
                "moderate" if a > 0.5 else
                "weak" if a > 0.2 else
                "near-zero")
    print(f"  ρ(RPS, Depth) = {sp_rd:+.2f}  [{_strength(sp_rd)}]: "
          f"the two milestones share signal but are not redundant; "
          f"reporting both is informative.")
    if sp_rc > 0.4:
        print(f"  ρ(RPS, −Cost) = {sp_rc:+.2f}: better RPS tends to come "
              f"from CHEAPER entries (algorithm > model, as already observed).")
    elif sp_rc < -0.4:
        print(f"  ρ(RPS, −Cost) = {sp_rc:+.2f}: better RPS requires more "
              f"compute spend.")
    else:
        print(f"  ρ(RPS, −Cost) = {sp_rc:+.2f}: cost and RPS are decoupled — "
              f"throwing money at the benchmark does not buy higher RPS.")


if __name__ == "__main__":
    main()
