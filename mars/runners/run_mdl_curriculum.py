"""Does the accumulating Library expand the reachability zone? (long-horizon)

The one principle predicts: building blocks discovered on earlier tasks are
"free" in bits on later tasks, so a growing Library should let the SAME weak
model solve more, and solve it FASTER (fewer rounds), as experience accumulates.

A/B over a sequence of episodes (seeds), same tasks, same weak model:
  A. no_transfer : each episode starts with an EMPTY library
  B. transfer    : one library PERSISTS across all episodes (accumulates)

Metrics:
  - exact rules / 5 (capability)
  - rounds-to-first-solve, averaged over solved rules (efficiency)
If transfer >= no_transfer on capability AND uses fewer rounds, the Library
expands reachability — the long-horizon claim.

Run:
  python -m mars.runners.run_mdl_curriculum --run_id mdl_curr_v1 \\
    --seeds 1,2,3,4,5 --difficulty easy --weak openai/gpt-4o-mini \\
    --rounds 6 --k 12 --overwrite
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
        load_dotenv(_env, override=False)
except ImportError:
    pass

from mars.mdl.engine import MDLEngine, Library, make_llm_proposer  # noqa: E402
from mars.mdl.tasks_seq import SeqRuleTask  # noqa: E402
from mars.runners.run_mdl_amplification import collect  # noqa: E402


def run_condition(seeds, difficulty, weak, rounds, k, transfer: bool):
    proposer = make_llm_proposer(weak)
    shared_lib = Library() if transfer else None
    per_seed = []
    for seed in seeds:
        slots = collect(seed, difficulty)
        lib = shared_lib if transfer else Library()
        engine = MDLEngine(proposer, max_rounds=rounds, k_per_round=k, library=lib)
        exact = 0
        solve_rounds = []
        for slot in range(1, 6):
            obs = slots[slot]
            if not obs:
                continue
            res = engine.compress(SeqRuleTask(slot, obs))
            solved = res.exact_rate >= 0.999
            exact += int(solved)
            if solved:
                # round at which the perfect program was first accepted
                r = next((a["round"] for a in res.accepted if a["exact_rate"] >= 0.999), res.rounds)
                solve_rounds.append(r)
        lib_size = len(engine.library.names())
        per_seed.append({"seed": seed, "exact": exact,
                         "avg_solve_round": round(sum(solve_rounds)/len(solve_rounds), 2) if solve_rounds else None,
                         "lib_size": lib_size})
        print(f"    seed {seed}: exact={exact}/5  "
              f"avg_solve_round={per_seed[-1]['avg_solve_round']}  lib={lib_size}")
    return per_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="mdl_curr_smoke")
    ap.add_argument("--seeds", default="1,2,3,4,5")
    ap.add_argument("--difficulty", default="easy")
    ap.add_argument("--weak", default="openai/gpt-4o-mini")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_dir = _PROJ / "lmw" / "mdl" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"exists; pass --overwrite: {out_path}")

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    print(f"=== Library transfer A/B (does accumulation expand reachability?) ===")
    print(f"weak={args.weak}  difficulty={args.difficulty}  seeds={seeds}\n")

    print("  [A] no_transfer (fresh library each episode):")
    a = run_condition(seeds, args.difficulty, args.weak, args.rounds, args.k, transfer=False)
    print("  [B] transfer (library accumulates across episodes):")
    b = run_condition(seeds, args.difficulty, args.weak, args.rounds, args.k, transfer=True)

    def agg(rows):
        n = len(rows)
        ex = sum(r["exact"] for r in rows) / n
        sr = [r["avg_solve_round"] for r in rows if r["avg_solve_round"] is not None]
        return ex, (sum(sr)/len(sr) if sr else None)
    ax, ar = agg(a)
    bx, br = agg(b)

    # Capability/efficiency over the LATER half (where accumulation has paid off)
    half = len(seeds) // 2
    bx_late, br_late = agg(b[half:]) if b[half:] else (bx, br)

    summary = {
        "run_id": args.run_id, "score_type": "library-transfer-ab",
        "weak": args.weak, "difficulty": args.difficulty, "seeds": seeds,
        "no_transfer": {"mean_exact": round(ax, 2), "mean_solve_round": _r(ar), "per_seed": a},
        "transfer": {"mean_exact": round(bx, 2), "mean_solve_round": _r(br), "per_seed": b},
        "transfer_late_half": {"mean_exact": round(bx_late, 2), "mean_solve_round": _r(br_late)},
        "reachability_expanded": bx >= ax and (br is not None and ar is not None and br <= ar),
    }
    out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"\n=== RESULT ===")
    print(f"  no_transfer : exact={ax:.2f}/5  avg_solve_round={_r(ar)}")
    print(f"  transfer    : exact={bx:.2f}/5  avg_solve_round={_r(br)}")
    print(f"  transfer (late half) : exact={bx_late:.2f}/5  avg_solve_round={_r(br_late)}")
    verdict = ("LIBRARY EXPANDS REACHABILITY" if summary["reachability_expanded"]
               else "no clear transfer effect")
    print(f"  -> {verdict}")
    print(f"summary -> {out_path}")


def _r(x):
    return round(x, 2) if x is not None else None


if __name__ == "__main__":
    main()
