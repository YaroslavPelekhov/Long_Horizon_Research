"""Ablation: disagreement-based pair selection vs random vs fixed defaults.

Research question
-----------------
Does adaptive pair selection by expected hypothesis disagreement lead to
faster rule convergence than random or fixed selection?

Three conditions (same grammar, same CPI scorer):
  A. pure_disagreement — at every step, compute disagreement across all valid
     candidate pairs and pick the one that maximises expected refutation.
  B. random — pick a random valid untried pair at each step.
  C. defaults — use the pre-specified default sequence [ABCDE/EDCBA, EDCBA/ABCDE, ...].
     This is the baseline used by the current system.

Metric: at each step t ∈ {1..max_steps}, the number of rule slots with
exact winner (loss_mean == 0.0). Higher faster = better.

Run
---
python -m mars.runners.run_selection_ablation \\
    --run_id ablation_easy_10seeds \\
    --difficulty easy --n_seeds 10 --max_steps 5 --overwrite

python -m mars.runners.run_selection_ablation \\
    --run_id ablation_hard_10seeds \\
    --difficulty hard --n_seeds 10 --max_steps 7 --overwrite
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import itertools
import json
import os
import random as _random_module
import sys
import time
from pathlib import Path
from typing import Any

_PROJ = Path(__file__).resolve().parent.parent.parent
_UH_REPO = _PROJ / "ultrahorizon_repo"
for _p in (_PROJ, _UH_REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from dotenv import load_dotenv
    _env = _PROJ / "autodiscovery" / ".env.local"
    if _env.exists():
        load_dotenv(_env, override=False)
except ImportError:
    pass

from mars.induction.cpi import disagreement_score  # noqa: E402
from mars.induction.uh_seq_inductor import (  # noqa: E402
    ALPHABET,
    UHSeqProgramInductor,
    _valid_sequence,
)


_DEFAULT_PAIRS = [
    ("ABCDE", "EDCBA"),
    ("EDCBA", "ABCDE"),
    ("AABCE", "DDEAC"),
    ("BAEDC", "CEBAD"),
    ("AACDE", "EABCD"),
    ("ABABA", "CDCDC"),
    ("BCDEA", "EADCB"),
]

_POOL: list[str] = [
    "".join(c)
    for c in itertools.islice(itertools.product(ALPHABET, repeat=5), 500)
    if _valid_sequence("".join(c))
]


def _run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def _exact_count(inductor: UHSeqProgramInductor) -> int:
    return sum(
        1
        for slot in range(1, 6)
        if inductor.ranked(slot)[0][1].loss_mean == 0.0
    )


def _select_random(inductor: UHSeqProgramInductor, rng: _random_module.Random) -> tuple[str, str]:
    candidates = [(m, v) for m in _POOL for v in _POOL if (m, v) not in inductor._tried_pairs and m != v]
    if not candidates:
        return _DEFAULT_PAIRS[0]
    return rng.choice(candidates[:200])


def _select_disagreement(inductor: UHSeqProgramInductor) -> tuple[str, str]:
    """Pure disagreement search from step 1 — no hardcoded defaults."""
    if not inductor.observations:
        return _DEFAULT_PAIRS[0]
    best_pair = None
    best_score = -1.0
    for main in _POOL[:40]:
        for vice in _POOL[:40]:
            if main == vice or (main, vice) in inductor._tried_pairs:
                continue
            preds = inductor._top_predictions_for_pair(main, vice, k=3)
            score = disagreement_score(preds)
            if score > best_score:
                best_score = score
                best_pair = (main, vice)
    return best_pair or _DEFAULT_PAIRS[0]


def _select_defaults(inductor: UHSeqProgramInductor, step_idx: int) -> tuple[str, str]:
    for pair in _DEFAULT_PAIRS:
        if pair not in inductor._tried_pairs:
            return pair
    return _POOL[step_idx % len(_POOL)], _POOL[(step_idx + 1) % len(_POOL)]


def run_one_seed(
    seed: int,
    difficulty_str: str,
    max_steps: int,
    selection_mode: str,
) -> dict[str, Any]:
    import numpy as np  # noqa: PLC0415
    from envs.common import Difficulty  # noqa: PLC0415
    from envs.seq_env.env import SequenceExploreEnvironment  # noqa: PLC0415

    rng = _random_module.Random(seed)
    _random_module.seed(seed)
    np.random.seed(seed)

    SequenceExploreEnvironment.load_judge_config = lambda _self: {}
    difficulty = getattr(Difficulty, difficulty_str.upper())

    with contextlib.redirect_stdout(io.StringIO()):
        env = SequenceExploreEnvironment(difficulty=difficulty, required_steps=max_steps, free=False)

    inductor = UHSeqProgramInductor()
    curve: list[int] = []  # exact_count at each step
    pairs_used: list[tuple[str, str]] = []
    t0 = time.time()

    for step_idx in range(max_steps):
        if selection_mode == "disagreement":
            main, vice = _select_disagreement(inductor)
        elif selection_mode == "random":
            main, vice = _select_random(inductor, rng)
        else:  # defaults
            main, vice = _select_defaults(inductor, step_idx)

        with contextlib.redirect_stdout(io.StringIO()):
            result = _run_async(env.input_sequences(main, vice))

        if not result.get("success"):
            break
        inductor.add_result(result)
        pairs_used.append((main, vice))
        curve.append(_exact_count(inductor))

    return {
        "seed": seed,
        "difficulty": difficulty_str,
        "mode": selection_mode,
        "max_steps": max_steps,
        "curve": curve,
        "final_exact": curve[-1] if curve else 0,
        "steps_to_5": next((i + 1 for i, e in enumerate(curve) if e == 5), None),
        "pairs_used": [list(p) for p in pairs_used],
        "wall_time_s": time.time() - t0,
    }


def run_ablation(args: argparse.Namespace) -> dict[str, Any]:
    modes = ["disagreement", "random", "defaults"]
    seeds = list(range(1, args.n_seeds + 1))
    all_runs: list[dict] = []

    for mode in modes:
        print(f"\n[ablation] mode={mode}")
        for seed in seeds:
            row = run_one_seed(seed, args.difficulty, args.max_steps, mode)
            all_runs.append(row)
            marker = "5/5" if row["final_exact"] == 5 else f"{row['final_exact']}/5"
            steps_to_5 = row["steps_to_5"] or "—"
            print(f"  seed={seed:2d}  final={marker}  steps_to_5={steps_to_5}  curve={row['curve']}")

    # Aggregate
    summary: dict[str, Any] = {
        "run_id": args.run_id,
        "score_type": "selection-ablation",
        "difficulty": args.difficulty,
        "n_seeds": args.n_seeds,
        "max_steps": args.max_steps,
        "modes": {},
    }

    for mode in modes:
        runs = [r for r in all_runs if r["mode"] == mode]
        mean_steps_to_5 = (
            sum(r["steps_to_5"] for r in runs if r["steps_to_5"] is not None)
            / max(1, sum(1 for r in runs if r["steps_to_5"] is not None))
        ) if any(r["steps_to_5"] for r in runs) else None
        frac_5_of_5 = sum(1 for r in runs if r["final_exact"] == 5) / len(runs)
        mean_final = sum(r["final_exact"] for r in runs) / len(runs)

        # Per-step mean exact
        step_means = []
        for step in range(1, args.max_steps + 1):
            vals = [r["curve"][step - 1] for r in runs if len(r["curve"]) >= step]
            step_means.append(sum(vals) / len(vals) if vals else 0.0)

        summary["modes"][mode] = {
            "frac_5_of_5": frac_5_of_5,
            "mean_final_exact": mean_final,
            "mean_steps_to_5": mean_steps_to_5,
            "step_means": step_means,
        }

    summary["runs"] = all_runs
    return summary


def _print_report(summary: dict[str, Any]) -> None:
    print("\n" + "=" * 65)
    print("SELECTION ABLATION RESULTS")
    print("=" * 65)
    print(f"difficulty={summary['difficulty']}  n_seeds={summary['n_seeds']}  max_steps={summary['max_steps']}")
    print()
    modes = list(summary["modes"])
    header = f"{'Step':<6}" + "".join(f"{m:<20}" for m in modes)
    print(header)
    print("-" * 65)
    max_steps = summary["max_steps"]
    for step in range(1, max_steps + 1):
        row = f"{step:<6}"
        for mode in modes:
            mean = summary["modes"][mode]["step_means"][step - 1]
            row += f"{mean:5.2f}/5                "
        print(row[:65])
    print("-" * 65)
    row = f"{'5/5%':<6}"
    for mode in modes:
        frac = summary["modes"][mode]["frac_5_of_5"]
        row += f"{frac*100:5.1f}%               "
    print(row[:65])
    row = f"{'Eff':<6}"
    for mode in modes:
        m5 = summary["modes"][mode]["mean_steps_to_5"]
        row += f"{'—' if m5 is None else f'{m5:.1f} steps':<20}"
    print(row[:65])
    print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", default=os.environ.get("MARS_ABLATION_RUN_ID", "ablation_smoke"))
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--n_seeds", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = _PROJ / "lmw" / "ablation" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"exists; pass --overwrite: {out_path}")

    summary = run_ablation(args)
    out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    _print_report(summary)
    print(f"summary → {out_path}")


if __name__ == "__main__":
    main()
