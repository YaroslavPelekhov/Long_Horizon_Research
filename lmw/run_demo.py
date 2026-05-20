"""
LMW v2 demo -- construct validity across a SCHEMA FAMILY.

DEV family   : several schema shapes (varying #clusters, chain length,
               #latents, decoys) -- the shapes used while building the metric.
HELD-OUT     : a schema whose SHAPE never appears in dev (4 clusters, chain
               length 3, 3 latents, regime shift ON). Fixtures/metric were
               never tuned on it. If the ablation ordering still holds here,
               the metric measures the capability, not a memorized shape.

Run:  python run_demo.py
"""

from __future__ import annotations

from harness import table_over
from scm import Schema

COLS = ["RPS", "axis1_agenda_ratio", "axis2_retention", "axis2_revision",
        "axis6_deadend_regret", "final_true_value"]


def _print(title, t) -> bool:
    print(f"\n=== {title} ===")
    head = f"{'variant':<10}" + "".join(f"{c:>21}" for c in COLS)
    print(head); print("-" * len(head))
    for name, m in t.items():
        row = f"{name:<10}"
        for c in COLS:
            v = m.get(c)
            row += f"{('' if v is None else v):>21}"
        print(row)
    full, mem = t["Full"]["RPS"], t["-mem"]["RPS"]
    goal, ab = t["-goal"]["RPS"], t["-abandon"]["RPS"]
    naive = max(t["Random"]["RPS"], t["GreedyObs"]["RPS"])
    print(f"\n  outer-loop value:  axis2(mem) {full-mem:+.3f}   "
          f"axis1(goal) {full-goal:+.3f}   axis6(abandon) {full-ab:+.3f}   "
          f"vs naive {full-naive:+.3f}")
    ok = full > mem > naive - 1e-9 and full > goal and full > ab and full > naive
    print(f"  self-check: {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    struct_seeds = [11, 22, 33]
    noise_seeds = [1, 2]

    dev_family = [
        Schema("d1", n_clusters=2, chain_len=1, n_latents=2, real_decoys=1),
        Schema("d2", n_clusters=3, chain_len=1, n_latents=2, real_decoys=1),
        Schema("d3", n_clusters=3, chain_len=2, n_latents=3, real_decoys=1),
        Schema("d4", n_clusters=4, chain_len=1, n_latents=2, real_decoys=2),
    ]
    heldout = [Schema("heldout", n_clusters=4, chain_len=3, n_latents=3,
                      real_decoys=2, regime_shift=True)]

    p1 = _print("DEV family (4 shapes x 3 struct x 2 noise = 24 worlds/variant)",
                table_over(dev_family, struct_seeds, noise_seeds))
    p2 = _print("HELD-OUT shape (4cl, chain3, 3lat, regime) -- UNTUNED",
                table_over(heldout, struct_seeds, noise_seeds))

    print("\n================ CONSTRUCT VALIDITY (schema family) ============")
    print(f"  DEV family self-check : {'PASS' if p1 else 'FAIL'}")
    print(f"  HELD-OUT   self-check : {'PASS' if p2 else 'FAIL'}  "
          f"<- the one that matters")
    print("  Verdict:", "metric generalizes across schema shapes"
          if p1 and p2 else "does NOT generalize across shapes")


if __name__ == "__main__":
    main()
