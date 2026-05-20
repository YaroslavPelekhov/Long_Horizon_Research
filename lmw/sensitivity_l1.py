"""
Reviewer point #2: is the L1 ablation ordering an artifact of a harsh
integrity penalty (i.e. the metric just rewards sandbagging)?

Sweep the integrity weight (scorer.FALSE_PENALTY) and re-run the deterministic
schema-family harness. If Full > {-mem,-goal,-abandon} > naive holds across a
wide range of the penalty weight, the L1 construct-validity claim is robust
and NOT an artifact of penalty tuning. Deterministic, no API.
"""

from __future__ import annotations

import scorer
from harness import table_over
from scm import Schema

LAMBDAS = [0.0, 0.5, 1.0, 1.5, 3.0, 5.0]
ORDER = ["Full", "-mem", "-goal", "-abandon"]


def main():
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    dev = [Schema("d1", n_clusters=2, chain_len=1, n_latents=2, real_decoys=1),
           Schema("d2", n_clusters=3, chain_len=1, n_latents=2, real_decoys=1),
           Schema("d3", n_clusters=3, chain_len=2, n_latents=3, real_decoys=1)]
    ss, ns = [11, 22], [1, 2]

    print("\n=== L1 RPS vs integrity weight (dev family, 3 schemas x 2x2 "
          "seeds) ===")
    print(f"{'FALSE_PENALTY':>13} | " + "".join(f"{v:>9}" for v in ORDER)
          + f"{'naive':>9} | ordering")
    print("-" * 76)
    base = scorer.FALSE_PENALTY
    stable = True
    for lam in LAMBDAS:
        scorer.FALSE_PENALTY = lam
        t = table_over(dev, ss, ns)
        rps = {k: t[k]["RPS"] for k in ORDER}
        naive = max(t["Random"]["RPS"], t["GreedyObs"]["RPS"])
        ok = (rps["Full"] > rps["-mem"] > naive
              and rps["Full"] > rps["-goal"]
              and rps["Full"] > rps["-abandon"]
              and min(rps["-mem"], rps["-goal"], rps["-abandon"]) is not None
              and rps["Full"] > naive)
        stable &= ok
        print(f"{lam:>13} | " + "".join(f"{rps[k]:>9.3f}" for k in ORDER)
              + f"{naive:>9.3f} | {'OK' if ok else 'BROKEN'}")
    scorer.FALSE_PENALTY = base
    print("-" * 76)
    print(f" ordering robust across ALL penalty weights: "
          f"{'YES (L1 construct validity is not a penalty artifact)' if stable else 'NO (L1 claim is penalty-sensitive -- report this)'}")


if __name__ == "__main__":
    main()
