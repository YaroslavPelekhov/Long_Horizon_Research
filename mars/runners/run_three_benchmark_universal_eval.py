"""Run the same MARS universal-evaluation protocol on three benchmarks.

This runner is intentionally a harness, not a new benchmark method.  Its job is
to prevent one-benchmark overfitting by forcing every claimed improvement to be
reported together across:

  - NewtonBench
  - DiscoveryBench
  - UltraHorizon

Each child runner remains the benchmark adapter.  This file only launches small
controlled slices, reads their summaries, and writes one combined report.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


_PROJ = Path(__file__).resolve().parent.parent.parent


def _run(cmd: list[str], *, cwd: Path) -> dict[str, Any]:
    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "wall_time_s": round(time.time() - started, 3),
        "output_tail": proc.stdout[-6000:],
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"missing": str(path)}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "path": str(path)}


def _newton_row(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "benchmark": "NewtonBench",
        "metric": "SA_all",
        "score": summary.get("SA_all"),
        "n": summary.get("n"),
        "answered": summary.get("answered"),
        "abstained": summary.get("abstained"),
        "details": {
            "modules": summary.get("modules"),
            "law_versions": summary.get("law_versions"),
            "methods": [
                {
                    "task": f"{r.get('module')}/{r.get('law_version')}",
                    "SA": r.get("SA"),
                    "method": r.get("method"),
                    "held_out_rmsle": r.get("held_out_rmsle", r.get("best_rmsle")),
                }
                for r in summary.get("rows", [])
            ],
        },
    }


def _discovery_row(summary: dict[str, Any]) -> dict[str, Any]:
    raw_score = summary.get("HMS_mean_100")
    consistency_score = summary.get("HMS_mean_consistency_100")
    return {
        "benchmark": "DiscoveryBench",
        "metric": "HMS_mean_100",
        "score": raw_score,
        "n": summary.get("n_tasks"),
        "answered": None,
        "abstained": None,
        "details": {
            "official_eval_skipped": raw_score is None,
            "discovery_modules": summary.get("discovery_modules"),
            "score_raw": raw_score,
            "score_consistency": consistency_score,
            "scores": summary.get("scores"),
            "consistency_scores": summary.get("consistency_scores"),
        },
    }


def _ultrahorizon_row(summary: dict[str, Any]) -> dict[str, Any]:
    score = None
    for key in ("mean_score", "mean_final_score", "score_mean", "avg_score", "final_score_mean"):
        if key in summary:
            score = summary.get(key)
            break
    if score is None:
        scores = summary.get("scores") or summary.get("final_scores")
        if isinstance(scores, list) and scores:
            try:
                score = sum(float(x) for x in scores) / len(scores)
            except Exception:
                score = None
    return {
        "benchmark": "UltraHorizon",
        "metric": "official_env_final_score",
        "score": score,
        "n": summary.get("n") or summary.get("n_tasks") or summary.get("episodes"),
        "answered": None,
        "abstained": None,
        "details": {
            "env": summary.get("env") or summary.get("envs"),
            "difficulty": summary.get("difficulty"),
            "ablation": summary.get("ablation"),
            "program_induction_score": summary.get("mean_program_induction_score"),
            "raw_summary_keys": sorted(summary.keys())[:32],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", default=f"threebench_universal_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--model", default="openai/gpt-4o-mini")
    parser.add_argument("--judge_model", default="openai/gpt-4o")
    parser.add_argument("--discovery_tasks", type=int, default=2)
    parser.add_argument("--discovery_task_keys", default="")
    parser.add_argument("--discovery_modules", default="")
    parser.add_argument("--discovery_official_eval", action="store_true")
    parser.add_argument("--uh_env", default="seq", choices=["grid", "seq", "bio", "all"])
    parser.add_argument("--uh_difficulty", default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--uh_steps", type=int, default=3)
    parser.add_argument("--uh_action_budget", type=int, default=12)
    parser.add_argument("--uh_seeds", default="42")
    parser.add_argument("--collect_only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = _PROJ / "lmw" / "three_benchmark_universal" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    if summary_path.exists() and not args.overwrite:
        raise SystemExit(f"exists; pass --overwrite: {summary_path}")

    child_runs = {
        "newton": f"{args.run_id}_newton",
        "discovery": f"{args.run_id}_discovery",
        "ultrahorizon": f"{args.run_id}_ultrahorizon",
    }

    commands = {
        "newton": [
            sys.executable,
            "-m",
            "mars.runners.run_nb_activeprobe",
            "--run_id",
            child_runs["newton"],
            "--model",
            args.model,
            "--modules",
            "m4_snell_law,m7_malus_law",
            "--difficulty",
            "easy",
            "--law_versions",
            "v0,v1",
            "--overwrite",
        ],
        "discovery": [
            sys.executable,
            "-m",
            "mars.runners.run_universal_discovery_real_eval",
            "--run_id",
            child_runs["discovery"],
            "--max_tasks",
            str(args.discovery_tasks),
            "--model",
            args.model,
            "--judge_model",
            args.judge_model,
            "--n_proposals",
            "0",
            "--max_rounds",
            "0",
            "--overwrite",
        ],
        "ultrahorizon": [
            sys.executable,
            "-m",
            "mars.runners.run_uh_official",
            "--run_id",
            child_runs["ultrahorizon"],
            "--env",
            args.uh_env,
            "--difficulty",
            args.uh_difficulty,
            "--steps",
            str(args.uh_steps),
            "--action_budget",
            str(args.uh_action_budget),
            "--seeds",
            args.uh_seeds,
            "--ablation",
            "MARS-full",
            "--generator_model",
            args.model,
            "--reflector_model",
            args.model,
            "--judge_model",
            args.judge_model,
            "--overwrite",
        ],
    }
    if args.discovery_modules:
        commands["discovery"][-1:-1] = ["--discovery_modules", args.discovery_modules]
    if args.discovery_task_keys:
        commands["discovery"][-1:-1] = ["--task_keys", *args.discovery_task_keys.split()]
    if not args.discovery_official_eval:
        commands["discovery"].insert(-1, "--skip_official_eval")

    paths = {
        "newton": _PROJ / "lmw" / "nb_activeprobe" / child_runs["newton"] / "summary.json",
        "discovery": _PROJ / "lmw" / "universal_discovery_real" / child_runs["discovery"] / "summary.json",
        "ultrahorizon": _PROJ / "lmw" / "uh_official" / child_runs["ultrahorizon"] / "summary.json",
    }
    runs: dict[str, Any] = {}
    if args.collect_only:
        runs = {name: {"cmd": cmd, "collect_only": True, "returncode": None} for name, cmd in commands.items()}
    else:
        for name, cmd in commands.items():
            print(f"=== running {name}: {' '.join(cmd)}", flush=True)
            runs[name] = _run(cmd, cwd=_PROJ)
            print(f"=== {name} returncode={runs[name]['returncode']} wall={runs[name]['wall_time_s']}s", flush=True)

    summaries = {name: _read_json(path) for name, path in paths.items()}
    table = [
        _newton_row(summaries["newton"]),
        _discovery_row(summaries["discovery"]),
        _ultrahorizon_row(summaries["ultrahorizon"]),
    ]
    result = {
        "run_id": args.run_id,
        "method_claim": "one universal MARS spine evaluated on three benchmarks; no per-result optimization inside this harness",
        "child_runs": child_runs,
        "paths": {k: str(v) for k, v in paths.items()},
        "commands": runs,
        "table": table,
        "summaries": summaries,
    }
    summary_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"summary → {summary_path}")
    for row in table:
        suffix = ""
        if row["benchmark"] == "DiscoveryBench":
            consistency = (row.get("details") or {}).get("score_consistency")
            suffix = f" consistency={consistency}" if consistency is not None else ""
        print(f"{row['benchmark']}: {row['metric']}={row['score']}{suffix} n={row['n']}")


if __name__ == "__main__":
    main()
