"""
Benchmark-correctness check: is the Open-Ended depth ordering an artifact of
the mastery-gate threshold tau?

Analog of sensitivity_l1.py (integrity weight) for the curriculum. If
Full/-mem reach >= -goal/-abandon/naive depth across a wide tau range, the
Open-Ended rung measures something stable; if the ordering flips with tau,
the depth numbers are arbitrary and the rung is not yet a valid benchmark.
Deterministic, no API.
"""

from __future__ import annotations

import statistics as st

from agents import ConnectorAgent, RandomAgent
from openworld import consistency_selftest, run_curriculum

SEEDS = [1, 2, 3, 4, 5]
TAUS = [0.05, 0.10, 0.15, 0.25, 0.40]
VARIANTS = {
    "Full":     (lambda: ConnectorAgent(True, True, True),  True),
    "-mem":     (lambda: ConnectorAgent(False, True, True),  False),
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
    consistency_selftest()
    print("\n=== Open-Ended depth vs mastery-gate tau (avg 5 seeds) ===")
    print(f"{'tau':>6} | " + "".join(f"{v:>10}" for v in VARIANTS)
          + " | depth-ordering OK?")
    print("-" * 78)
    stable = True
    for tau in TAUS:
        depth = {}
        for nm, (fac, persist) in VARIANTS.items():
            ds = [run_curriculum(fac, s, persist, tau=tau)["reached"]
                  for s in SEEDS]
            depth[nm] = st.fmean(ds)
        weak = max(depth["-goal"], depth["-abandon"], depth["naive"])
        # correctness: the competent process reaches strictly deeper than the
        # weak set, and actually progresses (depth > 0).
        ok = depth["Full"] > weak and depth["Full"] > 0.0
        stable &= ok
        print(f"{tau:>6} | " + "".join(f"{depth[v]:>10.2f}" for v in VARIANTS)
              + f" | {'OK' if ok else 'BROKEN'} (Full {depth['Full']:.2f} "
              f">= weak {weak:.2f})")
    print("-" * 78)
    print(f" depth ordering robust across tau in {TAUS}: "
          f"{'YES (Open-Ended rung is tau-stable)' if stable else 'NO (depth numbers are tau-arbitrary -- benchmark bug to fix)'}")


if __name__ == "__main__":
    main()
