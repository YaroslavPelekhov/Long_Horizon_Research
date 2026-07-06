"""Run the CPI prototype on the official UltraHorizon sequence environment."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import os
import random
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

try:
    from dotenv import load_dotenv

    _env_path = _PROJ / "autodiscovery" / ".env.local"
    if _env_path.exists():
        load_dotenv(_env_path, override=False)
except ImportError:
    pass

from mars.induction.uh_seq_inductor import UHSeqProgramInductor  # noqa: E402


def _run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def _judge_config(judge_model: str) -> dict[str, str]:
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "")
    if key.startswith("sk-or-") and not base_url:
        base_url = "https://openrouter.ai/api/v1"
    return {"model": judge_model, "base_url": base_url, "api_key": key}


def _extract_score(commit_result: dict[str, Any]) -> float:
    try:
        return float(commit_result["result"]["judge_result"]["final_score"])
    except Exception:
        return 0.0


def _extract_breakdown_sum(commit_result: dict[str, Any]) -> float | None:
    try:
        rows = commit_result["result"]["judge_result"]["score_breakdown"]
        return float(sum(float(row.get("awarded_score", 0.0)) for row in rows))
    except Exception:
        return None


def run_one(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np  # noqa: PLC0415
    from envs.common import Difficulty  # noqa: PLC0415
    from envs.seq_env.env import SequenceExploreEnvironment  # noqa: PLC0415

    random.seed(args.seed)
    np.random.seed(args.seed)
    SequenceExploreEnvironment.load_judge_config = lambda _self: _judge_config(args.judge_model)
    difficulty = getattr(Difficulty, args.difficulty.upper())

    with contextlib.redirect_stdout(io.StringIO()):
        env = SequenceExploreEnvironment(
            difficulty=difficulty,
            required_steps=args.steps,
            free=args.free,
        )

    inductor = UHSeqProgramInductor()
    tool_history: list[dict[str, Any]] = []
    t0 = time.time()

    for _ in range(args.steps):
        main, vice = inductor.select_next_pair()
        with contextlib.redirect_stdout(io.StringIO()):
            result = _run_async(env.input_sequences(main, vice))
        tool_history.append({"action": "input_sequences", "args": {"main": main, "vice": vice}, "result": result})
        if not result.get("success"):
            break
        inductor.add_result(result)

    report = inductor.build_report()
    induction_summary = inductor.summary()
    exact_program_fit = sum(
        1
        for candidates in induction_summary["rules"].values()
        if candidates and candidates[0]["loss_mean"] == 0
    )
    commit_result = None
    final_score = None
    if args.commit:
        with contextlib.redirect_stdout(io.StringIO()):
            commit_result = _run_async(env.commit_final_result(report))
        tool_history.append({"action": "commit_final_result", "args": {"content": report}, "result": commit_result})
        final_score = _extract_score(commit_result if isinstance(commit_result, dict) else {})
    breakdown_score = _extract_breakdown_sum(commit_result) if isinstance(commit_result, dict) else None

    return {
        "run_id": args.run_id,
        "score_type": "official-env-compatible" if args.commit else "dry-run-program-fit",
        "env": "seq",
        "difficulty": args.difficulty,
        "seed": args.seed,
        "steps": args.steps,
        "free": args.free,
        "judge_model": args.judge_model if args.commit else None,
        "committed": bool(args.commit and commit_result and commit_result.get("success")),
        "final_score": final_score,
        "breakdown_score": breakdown_score,
        "exact_program_fit": exact_program_fit,
        "wall_time_s": time.time() - t0,
        "induction": induction_summary,
        "submitted": report,
        "commit_result": commit_result,
        "tool_history": tool_history,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", default=os.environ.get("MARS_UH_SEQ_CPI_RUN_ID", "uh_seq_cpi_smoke"))
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="hard")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=7)
    parser.add_argument("--free", action="store_true")
    parser.add_argument("--commit", action="store_true")
    parser.add_argument("--judge_model", default=os.environ.get("MARS_UH_OFFICIAL_JUDGE_MODEL", "openai/gpt-4o"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = _PROJ / "lmw" / "uh_seq_cpi" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"summary exists; pass --overwrite: {out_path}")

    row = run_one(args)
    out_path.write_text(json.dumps(row, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print("=== UH Seq CPI ===")
    print(f"run_id={args.run_id} difficulty={args.difficulty} steps={args.steps} commit={args.commit}")
    print(f"program_fit_rules={row['exact_program_fit']}/5 committed={row['committed']} score={row['final_score']}")
    print(f"summary={out_path}")
    print("\n" + row["submitted"])


if __name__ == "__main__":
    main()
