"""
NewtonBench sweep with Fit-then-Snap exact-symbolic recovery.

Quantifies the wall-break: StructuralSketchSolver (LLM proposes structure) +
curve_fit (numeric holes) + Fit-then-Snap (snap exponents to exact rationals)
→ recover the EXACT symbolic law → flip binary SA from 0 to 1.

Baseline (RASC-fit / regression) SA by tier: easy ~1.0, medium ~0.33, hard 0.0.

Run:  python -m mars.srhp.nb_snap_sweep
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
_PROJ = Path(__file__).resolve().parent.parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

try:
    from dotenv import load_dotenv
    _env = _PROJ / "autodiscovery" / ".env.local"
    if _env.exists():
        load_dotenv(_env, override=False)
except ImportError:
    pass

import statistics as st

from mars.adapters.newtonbench_adapter import NBTask, NewtonBenchAdapter
from mars.srhp.sketch_solver import StructuralSketchSolver


def solve_task(mod, diff, n_data=22, rounds=5):
    adapter = NewtonBenchAdapter(task=NBTask(mod, diff, "v0"), budget=60, judge_model="gpt41")
    cands = adapter.srhp_candidate_experiments(48)
    inputs, outputs = [], []
    step = max(1, len(cands) // n_data)
    for exp in cands[::step][:n_data]:
        y = adapter.srhp_run(exp)
        if y == y:
            inputs.append(exp); outputs.append(float(y))
    solver = StructuralSketchSolver(model="openai/gpt-4o-mini", max_rounds=rounds)
    res = solver.solve(inputs, outputs)
    if not res:
        return None, 0.0, 0.0
    adapter.srhp_finalize(res.to_law_code(), "discovered_law")
    sc = adapter.score_episode([])
    return res, float(sc.get("SA", 0.0)), float(sc.get("numerical_accuracy", 0.0))


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    mods = os.environ.get(
        "MARS_NB_MODULES",
        "m0_gravity,m1_coulomb_force,m9_hooke_law,m5_radioactive_decay,m8_sound_speed,m3_fourier_law"
    ).split(",")
    diffs = os.environ.get("MARS_NB_DIFFS", "easy,medium,hard").split(",")

    print("\n=== NewtonBench × Fit-then-Snap (exact-symbolic recovery) ===")
    print(f"  modules={mods}\n  baseline RASC-fit SA: easy~1.0 medium~0.33 hard~0.0\n")

    by_diff = {d.strip(): [] for d in diffs}
    by_diff_na = {d.strip(): [] for d in diffs}
    for m in mods:
        for d in diffs:
            m, d = m.strip(), d.strip()
            try:
                res, sa, na = solve_task(m, d)
                expr = res.expr[:55] if res else "None"
            except Exception as e:
                sa, na, expr = 0.0, 0.0, f"err {str(e)[:40]}"
            by_diff[d].append(sa); by_diff_na[d].append(na)
            flag = " <== SA FLIP" if sa >= 1 else ""
            print(f"  {m:<20}/{d:<7} SA={sa:.1f} num_acc={na:.2f}  {expr}{flag}", flush=True)

    print("\n=== Snap sweep summary (SA.mean / num_acc.mean by tier) ===")
    for d in by_diff:
        sa = by_diff[d]; na = by_diff_na[d]
        if sa:
            print(f"  {d:<8} SA.mean={st.fmean(sa):.3f}  num_acc.mean={st.fmean(na):.3f}  "
                  f"(flips: {sum(1 for x in sa if x>=1)}/{len(sa)})")
    allsa = [x for v in by_diff.values() for x in v]
    print(f"  OVERALL SA.mean={st.fmean(allsa):.3f}  (vs RASC-fit ~0.44)")


if __name__ == "__main__":
    main()
