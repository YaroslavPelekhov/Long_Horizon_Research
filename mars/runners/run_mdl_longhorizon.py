"""Long-horizon reachability curve: does the library keep expanding what the
weak model can solve, episode after episode?

One persisting Library over N episodes. We record the per-episode capability
(exact rules / 5) and fit its trend. A positive slope = the reachability zone
keeps expanding as the compression library grows. We also report a cold baseline
(empty library) averaged over the same episodes for reference.

Writes the curve incrementally (flush) so progress is visible.

Run:
  python -m mars.runners.run_mdl_longhorizon --run_id mdl_long_v1 \\
    --episodes 14 --difficulty easy --weak openai/gpt-4o-mini \\
    --rounds 4 --k 10 --overwrite
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_PROJ = Path(__file__).resolve().parent.parent.parent
for _p in (_PROJ, _PROJ / "ultrahorizon_repo"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from dotenv import load_dotenv
    _env = _PROJ / "autodiscovery" / ".env.local"
    if _env.exists():
        load_dotenv(_env, override=True)
except ImportError:
    pass

from mars.mdl.engine import MDLEngine, Library, make_llm_proposer  # noqa: E402
from mars.mdl.tasks_seq import SeqRuleTask  # noqa: E402
from mars.runners.run_mdl_amplification import collect  # noqa: E402


def _slope(ys: list[float]) -> float:
    n = len(ys)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mx = sum(xs) / n; my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="mdl_long_smoke")
    ap.add_argument("--episodes", type=int, default=14)
    ap.add_argument("--difficulty", default="easy")
    ap.add_argument("--weak", default="openai/gpt-4o-mini")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_dir = _PROJ / "lmw" / "mdl" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"exists; pass --overwrite: {out_path}")

    proposer = make_llm_proposer(args.weak)
    shared = Library()                       # ONE library persists across episodes
    curve = []          # exact/5 per episode (transfer)
    cold = []           # exact/5 per episode (fresh library, baseline)
    lib_sizes = []
    print(f"=== Long-horizon reachability curve (weak={args.weak}) ===", flush=True)
    print(f"episodes={args.episodes} difficulty={args.difficulty} (transfer vs cold)\n", flush=True)

    for ep in range(args.episodes):
        seed = 100 + ep                      # fresh episodes not seen before
        slots = collect(seed, args.difficulty)

        # transfer: persisting library
        eng_t = MDLEngine(proposer, max_rounds=args.rounds, k_per_round=args.k, library=shared)
        ex_t = 0
        for slot in range(1, 6):
            if slots[slot]:
                if eng_t.compress(SeqRuleTask(slot, slots[slot])).exact_rate >= 0.999:
                    ex_t += 1
        curve.append(ex_t)
        lib_sizes.append(len(shared.names()))

        # cold: fresh library, same episode
        eng_c = MDLEngine(proposer, max_rounds=args.rounds, k_per_round=args.k, library=Library())
        ex_c = 0
        for slot in range(1, 6):
            if slots[slot]:
                if eng_c.compress(SeqRuleTask(slot, slots[slot])).exact_rate >= 0.999:
                    ex_c += 1
        cold.append(ex_c)

        print(f"  ep {ep:2d} (seed {seed}): transfer={ex_t}/5  cold={ex_c}/5  "
              f"lib={lib_sizes[-1]}", flush=True)
        # incremental save
        out_path.write_text(json.dumps({
            "run_id": args.run_id, "weak": args.weak, "difficulty": args.difficulty,
            "episodes_done": ep + 1, "transfer_curve": curve, "cold_curve": cold,
            "lib_sizes": lib_sizes,
        }, indent=2), encoding="utf-8")

    n = len(curve)
    early = sum(curve[: max(1, n // 3)]) / max(1, n // 3)
    late = sum(curve[-max(1, n // 3):]) / max(1, n // 3)
    summary = {
        "run_id": args.run_id, "score_type": "longhorizon-reachability-curve",
        "weak": args.weak, "difficulty": args.difficulty, "episodes": n,
        "transfer_curve": curve, "cold_curve": cold, "lib_sizes": lib_sizes,
        "transfer_slope": round(_slope([float(x) for x in curve]), 3),
        "cold_slope": round(_slope([float(x) for x in cold]), 3),
        "transfer_early_mean": round(early, 2), "transfer_late_mean": round(late, 2),
        "cold_mean": round(sum(cold) / n, 2), "transfer_mean": round(sum(curve) / n, 2),
        "final_lib_size": lib_sizes[-1] if lib_sizes else 0,
        "reachability_expands": _slope([float(x) for x in curve]) > 0 and late >= early,
    }
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n=== REACHABILITY CURVE ===", flush=True)
    print(f"  transfer curve : {curve}", flush=True)
    print(f"  cold curve     : {cold}", flush=True)
    print(f"  transfer slope : {summary['transfer_slope']:+.3f}/episode   "
          f"cold slope: {summary['cold_slope']:+.3f}", flush=True)
    print(f"  transfer early->late : {summary['transfer_early_mean']} -> {summary['transfer_late_mean']}", flush=True)
    print(f"  transfer_mean={summary['transfer_mean']}  cold_mean={summary['cold_mean']}  "
          f"final_lib={summary['final_lib_size']}", flush=True)
    print(f"  -> {'REACHABILITY EXPANDS WITH LIBRARY' if summary['reachability_expands'] else 'flat'}", flush=True)
    print(f"summary -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
