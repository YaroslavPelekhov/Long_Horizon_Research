"""Bio-CPI runner for UltraHorizon Alien Genetics Lab.

Uses the official GeneticsLabEnvironment with systematic genetics protocol
(Bio-CPI) instead of a generic LLM agent. No rubric leakage — rules are
discovered from experimental data.

Usage
-----
python -m mars.runners.run_uh_bio_cpi \\
    --run_id bio_cpi_s42_gpt4o \\
    --seeds 42,43,44,45,46 --budget 18 \\
    --judge_model openai/gpt-4o --overwrite
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import os
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

from mars.induction.bio_cpi import BioCPI  # noqa: E402


def _run_async(coro: Any) -> Any:
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
    return {"model": judge_model, "api_key": key, "base_url": base_url}


def _extract_score(commit_result: dict[str, Any]) -> float:
    try:
        jr = commit_result.get("judge_result", commit_result.get("result", {}).get("judge_result", {}))
        return float(jr.get("final_score", 0.0))
    except Exception:
        return 0.0


def run_one_seed(seed: int, budget: int, judge_model: str) -> dict[str, Any]:
    from envs.bio_env.env import GeneticsLabEnvironment  # noqa: PLC0415
    from envs.common import Difficulty  # noqa: PLC0415

    # Patch judge config
    GeneticsLabEnvironment.load_judge_config = lambda self: _judge_config(judge_model)

    with contextlib.redirect_stdout(io.StringIO()):
        env = GeneticsLabEnvironment(
            seed=seed,
            difficulty=Difficulty.HARD,
            required_steps=budget,
        )

    t0 = time.time()
    cpi = BioCPI(env, budget=budget)

    async def _run() -> dict[str, Any]:
        report = await cpi.run_protocol()
        # Commit to official judge
        commit = await env.commit_final_result(report)
        return {"report": report, "commit": commit}

    result = _run_async(_run())
    score = _extract_score(result["commit"])
    wall = time.time() - t0

    return {
        "seed": seed,
        "budget": budget,
        "judge_model": judge_model,
        "final_score": score,
        "steps_used": cpi.steps_used,
        "cross_count": cpi.cross_count,
        "n_crosses": len(cpi.crosses),
        "n_organisms": len(cpi.organisms),
        "ploidy_inferred": cpi.hyp.ploidy,
        "color_order": cpi.hyp.color_dominant,
        "shell_mechanism": cpi.hyp.shell_mechanism,
        "size_mechanism": cpi.hyp.size_mechanism,
        "lethal_detected": cpi.hyp.lethal_combo is not None,
        "viability_rates": cpi.hyp.viability_by_cross,
        "report_len": len(result["report"]),
        "commit_result": result["commit"],
        "wall_time_s": wall,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", default=os.environ.get("MARS_BIO_CPI_RUN_ID", "bio_cpi_smoke"))
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--budget", type=int, default=18)
    parser.add_argument("--judge_model", default=os.environ.get("MARS_JUDGE_MODEL", "openai/gpt-4o"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = _PROJ / "lmw" / "bio_cpi" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"exists; pass --overwrite: {out_path}")

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    results = []
    scores = []

    print(f"Bio-CPI: {len(seeds)} seeds, budget={args.budget}, judge={args.judge_model}")
    print()

    for seed in seeds:
        print(f"  seed={seed} ...", end=" ", flush=True)
        try:
            row = run_one_seed(seed, args.budget, args.judge_model)
        except Exception as e:
            row = {"seed": seed, "final_score": 0.0, "error": str(e)}
        results.append(row)
        score = row.get("final_score", 0.0)
        scores.append(score)
        steps = row.get("steps_used", "?")
        crosses = row.get("n_crosses", "?")
        lethal = "lethal✓" if row.get("lethal_detected") else "lethal?"
        print(f"score={score:.1f}/100  steps={steps}  crosses={crosses}  {lethal}")

    mean_score = sum(scores) / len(scores) if scores else 0.0
    summary = {
        "run_id": args.run_id,
        "score_type": "official-env-compatible",
        "judge_model": args.judge_model,
        "budget": args.budget,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "mean_score": mean_score,
        "sd_score": (
            (sum((s - mean_score) ** 2 for s in scores) / len(scores)) ** 0.5
            if len(scores) > 1 else 0.0
        ),
        "scores": scores,
        "results": results,
    }

    out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print()
    print(f"=== Bio-CPI Results ===")
    print(f"N={len(seeds)}  mean={mean_score:.1f}/100  scores={scores}")
    print(f"Compare: dev_proxy_mean=87.0  Gemini-2.5-Pro=14.33 (full UH benchmark)")
    print(f"summary → {out_path}")


if __name__ == "__main__":
    main()
