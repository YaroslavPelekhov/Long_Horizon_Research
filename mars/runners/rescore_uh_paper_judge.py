"""Rescore UltraHorizon submissions with the paper-style DeepSeek-R1 judge.

This script does not rerun the agent. It reuses saved terminal submissions,
instantiates the official UltraHorizon environment, and calls its own
`commit_final_result` scoring path with the paper judge label/config used by
`run_uh_official.py`.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import contextlib
import io
import json
import statistics as st_stats
import sys
import time
from pathlib import Path
from typing import Any

_PROJ = Path(__file__).resolve().parent.parent.parent
_UH_REPO = _PROJ / "ultrahorizon_repo"
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))
if str(_UH_REPO) not in sys.path:
    sys.path.insert(0, str(_UH_REPO))

from mars.runners.run_uh_official import (
    PAPER_JUDGE_LABEL,
    PAPER_JUDGE_OPENROUTER_MODEL,
    UHTask,
    UltraHorizonOfficialAdapter,
)


def _extract_submission(row: dict[str, Any]) -> str:
    submitted = str(row.get("submitted") or "").strip()
    if submitted:
        return submitted
    for item in row.get("env_tool_history", []) or []:
        if item.get("action") == "commit_final_result":
            content = ((item.get("args") or {}).get("content") or "").strip()
            if content:
                return str(content)
    return str(row.get("submitted_preview") or "").strip()


def _rescore_row(row: dict[str, Any], *, paper_judge_api_model: str) -> dict[str, Any]:
    content = _extract_submission(row)
    env_name = str(row.get("env"))
    seed = int(row.get("seed", 42))
    steps = int(row.get("steps", 50))
    difficulty = str(row.get("difficulty", "hard"))
    task = UHTask(
        env_name=env_name,
        seed=seed,
        steps=steps,
        free=True,
        difficulty=difficulty,
        judge_model=PAPER_JUDGE_LABEL,
        action_budget=1,
        use_env_hints=False,
        enable_fallback_commit=False,
        paper_style_judge=True,
        paper_judge_label=PAPER_JUDGE_LABEL,
        paper_judge_api_model=paper_judge_api_model,
    )
    t0 = time.time()
    adapter = UltraHorizonOfficialAdapter(task)
    with contextlib.redirect_stdout(io.StringIO()):
        result = adapter.execute("commit_final_result", {"content": content}).raw
    score = adapter.score_episode([]).get("final_score", 0.0)
    scored = adapter.score_episode([])
    return {
        "benchmark": "UltraHorizon",
        "env": env_name,
        "seed": seed,
        "difficulty": difficulty,
        "steps": steps,
        "free": False,
        "source_task_label": row.get("task_label"),
        "source_final_score": row.get("final_score"),
        "source_judge_model": row.get("judge_model"),
        "judge_model": PAPER_JUDGE_LABEL,
        "paper_style_judge": True,
        "paper_judge_api_model": paper_judge_api_model,
        "submitted": content,
        "submitted_chars": len(content),
        "final_score": score,
        "committed": scored.get("committed", False),
        "judge_result": scored.get("judge_result", {}),
        "raw_commit_result": result,
        "wall_time_s": time.time() - t0,
    }


def _error_row(row: dict[str, Any], paper_judge_api_model: str, error: BaseException) -> dict[str, Any]:
    return {
        "benchmark": "UltraHorizon",
        "env": row.get("env"),
        "seed": row.get("seed"),
        "source_final_score": row.get("final_score"),
        "judge_model": PAPER_JUDGE_LABEL,
        "paper_style_judge": True,
        "paper_judge_api_model": paper_judge_api_model,
        "final_score": 0.0,
        "committed": False,
        "error": f"{type(error).__name__}: {error}",
    }


def _parse_envs(value: str) -> set[str] | None:
    if value == "all":
        return None
    return {x.strip() for x in value.split(",") if x.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_run_jsonl", required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--envs", default="all", help="all or comma-separated grid,seq,bio")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--paper_judge_api_model", default=PAPER_JUDGE_OPENROUTER_MODEL)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    in_path = Path(args.input_run_jsonl)
    if not in_path.is_absolute():
        in_path = _PROJ / in_path
    rows = [json.loads(line) for line in in_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    env_filter = _parse_envs(args.envs)
    if env_filter is not None:
        rows = [r for r in rows if r.get("env") in env_filter]
    if args.limit:
        rows = rows[: args.limit]

    out_dir = _PROJ / "lmw" / "uh_official" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / "run.jsonl"
    summary_path = out_dir / "summary.json"
    if out_jsonl.exists() and not args.overwrite:
        raise SystemExit(f"run log exists; pass --overwrite: {out_jsonl}")
    if args.overwrite and out_jsonl.exists():
        out_jsonl.unlink()

    print("\n=== UltraHorizon paper-style judge rescore ===")
    print(f"  input={in_path}")
    print(f"  run_id={args.run_id}")
    print(f"  envs={args.envs} N={len(rows)}")
    print(f"  judge_label={PAPER_JUDGE_LABEL} api_model={args.paper_judge_api_model}\n")

    out_rows: list[dict[str, Any]] = []
    t_start = time.time()
    with out_jsonl.open("w", encoding="utf-8") as f:
        if args.workers <= 1:
            iterator = []
            for i, row in enumerate(rows, 1):
                print(f"  [{i}/{len(rows)}] {row.get('env')} seed={row.get('seed')} ...", flush=True)
                iterator.append((i, row, None))
            for i, row, _ in iterator:
                try:
                    scored = _rescore_row(row, paper_judge_api_model=args.paper_judge_api_model)
                except Exception as e:
                    scored = _error_row(row, args.paper_judge_api_model, e)
                out_rows.append(scored)
                f.write(json.dumps(scored, ensure_ascii=False, default=str) + "\n")
                f.flush()
                print(f"      [{i}/{len(rows)}] score={float(scored.get('final_score', 0.0)):.1f}", flush=True)
        else:
            print(f"  workers={args.workers}", flush=True)
            with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
                future_to_meta = {
                    ex.submit(_rescore_row, row, paper_judge_api_model=args.paper_judge_api_model): (i, row)
                    for i, row in enumerate(rows, 1)
                }
                done_count = 0
                for fut in cf.as_completed(future_to_meta):
                    i, row = future_to_meta[fut]
                    try:
                        scored = fut.result()
                    except Exception as e:
                        scored = _error_row(row, args.paper_judge_api_model, e)
                    done_count += 1
                    out_rows.append(scored)
                    f.write(json.dumps(scored, ensure_ascii=False, default=str) + "\n")
                    f.flush()
                    print(
                        f"      done={done_count}/{len(rows)} "
                        f"row={i} env={row.get('env')} seed={row.get('seed')} "
                        f"score={float(scored.get('final_score', 0.0)):.1f}",
                        flush=True,
                    )

    scores = [float(r.get("final_score", 0.0)) for r in out_rows]
    per_env = {}
    for env in sorted({r.get("env") for r in out_rows}):
        env_scores = [float(r.get("final_score", 0.0)) for r in out_rows if r.get("env") == env]
        per_env[str(env)] = {
            "n": len(env_scores),
            "mean_score": st_stats.fmean(env_scores) if env_scores else 0.0,
            "sd_score": st_stats.stdev(env_scores) if len(env_scores) > 1 else 0.0,
            "exact_100": sum(1 for s in env_scores if s >= 100.0),
        }
    summary = {
        "run_id": args.run_id,
        "benchmark": "UltraHorizon",
        "score_type": "official-env-compatible-paper-style-judge-rescore",
        "source_run_jsonl": str(in_path),
        "exact_paper_judge": True,
        "paper_style_judge": True,
        "paper_judge_label": PAPER_JUDGE_LABEL,
        "paper_judge_api_model": args.paper_judge_api_model,
        "n": len(out_rows),
        "mean_score": st_stats.fmean(scores) if scores else 0.0,
        "sd_score": st_stats.stdev(scores) if len(scores) > 1 else 0.0,
        "exact_100": sum(1 for s in scores if s >= 100.0),
        "exact_rate": (sum(1 for s in scores if s >= 100.0) / len(scores)) if scores else 0.0,
        "n_committed": sum(1 for r in out_rows if r.get("committed")),
        "per_env": per_env,
        "wall_time_s": time.time() - t_start,
        "paths": {"run_jsonl": str(out_jsonl), "summary": str(summary_path)},
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== Summary ===")
    print(f"  mean_score={summary['mean_score']:.2f} N={summary['n']} exact={summary['exact_100']}")
    print(f"  per_env={json.dumps(per_env, ensure_ascii=False)}")
    print(f"  summary={summary_path}")


if __name__ == "__main__":
    main()
