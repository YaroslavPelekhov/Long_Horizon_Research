"""Sweep the UltraHorizon Seq CPI prototype over seeds/difficulties."""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

from mars.runners.run_uh_seq_cpi import run_one

_PROJ = Path(__file__).resolve().parent.parent.parent


def _parse_csv(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _has_judge_key() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", default=os.environ.get("MARS_UH_SEQ_CPI_SWEEP_ID", "uh_seq_cpi_sweep"))
    parser.add_argument("--difficulties", default="easy,hard")
    parser.add_argument("--seeds", default="1,2,3,4,5")
    parser.add_argument("--easy_steps", type=int, default=5)
    parser.add_argument("--hard_steps", type=int, default=7)
    parser.add_argument("--free", action="store_true")
    parser.add_argument("--commit", action="store_true")
    parser.add_argument("--judge_model", default=os.environ.get("MARS_UH_OFFICIAL_JUDGE_MODEL", "openai/gpt-4o"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.commit and not _has_judge_key():
        raise SystemExit(
            "commit requested, but neither OPENAI_API_KEY nor OPENROUTER_API_KEY is set in the environment"
        )

    difficulties = _parse_csv(args.difficulties)
    seeds = [int(x) for x in _parse_csv(args.seeds)]
    out_dir = _PROJ / "lmw" / "uh_seq_cpi" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    run_log = out_dir / "run.jsonl"
    summary_path = out_dir / "summary.json"
    if run_log.exists() and not args.overwrite:
        raise SystemExit(f"run log exists; pass --overwrite: {run_log}")
    if args.overwrite and run_log.exists():
        run_log.unlink()

    rows: list[dict[str, Any]] = []
    t0 = time.time()
    with run_log.open("w", encoding="utf-8") as f:
        for difficulty in difficulties:
            steps = args.easy_steps if difficulty == "easy" else args.hard_steps
            for seed in seeds:
                child_args = Namespace(
                    run_id=f"{args.run_id}_{difficulty}_s{seed}",
                    difficulty=difficulty,
                    seed=seed,
                    steps=steps,
                    free=args.free,
                    commit=args.commit,
                    judge_model=args.judge_model,
                    overwrite=True,
                )
                row = run_one(child_args)
                rows.append(row)
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                f.flush()
                print(
                    f"{difficulty} seed={seed} exact={row['exact_program_fit']}/5 "
                    f"committed={row['committed']} score={row['final_score']}",
                    flush=True,
                )

    exact = [int(r.get("exact_program_fit", 0)) for r in rows]
    scores = [
        float(r["final_score"])
        for r in rows
        if r.get("final_score") is not None
    ]
    by_diff: dict[str, dict[str, Any]] = {}
    for difficulty in difficulties:
        subset = [r for r in rows if r["difficulty"] == difficulty]
        vals = [int(r.get("exact_program_fit", 0)) for r in subset]
        diff_scores = [
            float(r["final_score"])
            for r in subset
            if r.get("final_score") is not None
        ]
        by_diff[difficulty] = {
            "n": len(subset),
            "mean_exact_program_fit": st.fmean(vals) if vals else 0.0,
            "all_exact": all(v == 5 for v in vals) if vals else False,
            "mean_final_score": st.fmean(diff_scores) if diff_scores else None,
        }

    summary = {
        "run_id": args.run_id,
        "score_type": "official-env-compatible" if args.commit else "dry-run-program-fit",
        "difficulties": difficulties,
        "seeds": seeds,
        "n": len(rows),
        "commit": args.commit,
        "judge_model": args.judge_model if args.commit else None,
        "mean_exact_program_fit": st.fmean(exact) if exact else 0.0,
        "all_exact": all(v == 5 for v in exact) if exact else False,
        "mean_final_score": st.fmean(scores) if scores else None,
        "by_difficulty": by_diff,
        "wall_time_s": time.time() - t0,
        "paths": {"run_log": str(run_log), "summary": str(summary_path)},
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== UH Seq CPI sweep summary ===")
    print(f"n={summary['n']} all_exact={summary['all_exact']} mean_exact={summary['mean_exact_program_fit']:.2f}/5")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
