"""Universal CPI suite — ONE engine, 4 benchmarks, fully autonomous.

This is the central universality + autonomy demonstration. A single
UniversalCPI engine is instantiated once and run against four heterogeneous
benchmarks via thin adapters. No adapter contains a domain answer or the
scoring rubric. Every hypothesis is discovered through:

    LLM proposal → sandbox validation → refutation on held-out observations
    → MDL selection → generic verbalization

Run:
  python -m mars.runners.run_universal_cpi_suite --run_id suite_v1 \\
      --benchmarks uh_seq,newtonbench,uh_bio,discoverybench \\
      --model openai/gpt-4o --overwrite
"""

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
for _p in (_PROJ, _PROJ / "ultrahorizon_repo", _PROJ / "newtonbench_repo", _PROJ / "discoverybench_repo"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from dotenv import load_dotenv
    _env = _PROJ / "autodiscovery" / ".env.local"
    if _env.exists():
        load_dotenv(_env, override=False)
except ImportError:
    pass

from mars.induction.universal_cpi import UniversalCPI  # noqa: E402
from mars.induction.cpi_adapters import (  # noqa: E402
    BioAdapter,
    DiscoveryAdapter,
    NewtonAdapter,
    UHSeqAdapter,
)
from mars.skills.supermetrics import aggregate_supermetrics  # noqa: E402


def _run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def _judge_cfg(model: str) -> dict[str, str]:
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY", "")
    base = os.environ.get("OPENAI_BASE_URL", "")
    if key.startswith("sk-or-") and not base:
        base = "https://openrouter.ai/api/v1"
    return {"model": model, "api_key": key, "base_url": base}


def _aggregate_suite_supermetrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    agg = aggregate_supermetrics(rows)
    return {
        "universal_score": agg.get("mean_universal_score", 0.0),
        "aggregate": agg,
    }


def _supermetric_score(row: dict[str, Any]) -> float:
    sm = row.get("supermetrics") or {}
    if "universal_score" in sm:
        return float(sm.get("universal_score") or 0.0)
    return float(sm.get("mean_universal_score") or 0.0)


# ===========================================================================
# Benchmark drivers — each collects data, then calls the SHARED engine.
# ===========================================================================

def run_uh_seq(engine: UniversalCPI, seed: int, judge_model: str) -> dict[str, Any]:
    import numpy as np
    from envs.common import Difficulty
    from envs.seq_env.env import SequenceExploreEnvironment

    random.seed(seed); np.random.seed(seed)
    SequenceExploreEnvironment.load_judge_config = lambda s: {}
    with contextlib.redirect_stdout(io.StringIO()):
        env = SequenceExploreEnvironment(difficulty=Difficulty.EASY, required_steps=5, free=False)

    defaults = [("ABCDE","EDCBA"),("EDCBA","ABCDE"),("AABCE","DDEAC"),("BAEDC","CEBAD"),("AACDE","EABCD")]
    raw = []
    async def collect():
        for m, v in defaults:
            with contextlib.redirect_stdout(io.StringIO()):
                r = await env.input_sequences(m, v)
            if r.get("success"):
                raw.append(r)
    _run_async(collect())

    exact = 0
    per_rule = []
    supermetric_rows = []
    for slot in range(1, 6):
        adapter = UHSeqAdapter(env, rule_slot=slot)
        adapter.set_observations(raw)
        res = engine.run(adapter)
        supermetric_rows.append(res.supermetrics)
        best = res.winners[0] if res.winners else None
        loss = best[1].loss_mean if best else 1.0
        is_exact = loss == 0.0
        exact += int(is_exact)
        per_rule.append({"slot": slot, "exact": is_exact, "loss": loss,
                         "winner": best[0].name if best else None, "n_valid": res.n_valid,
                         "supermetrics": res.supermetrics})
    return {
        "benchmark": "uh_seq", "metric": "exact_rules", "score": exact, "max": 5,
        "normalized": exact / 5, "detail": per_rule,
        "supermetrics": _aggregate_suite_supermetrics(supermetric_rows),
    }


def run_newtonbench(engine: UniversalCPI, seed: int, judge_model: str) -> dict[str, Any]:
    import importlib
    from mars.induction.nb_cpi import make_probe_inputs, recover_force

    module = importlib.import_module("modules.m0_gravity")
    signature = str(module.FUNCTION_SIGNATURE).strip()
    params = [p.strip() for p in signature[signature.index("(")+1:signature.rindex(")")].split(",") if p.strip()]

    data_points = []
    for inp in make_probe_inputs(params, system="vanilla_equation"):
        raw = module.run_experiment_for_module(noise_level=0.0, difficulty="easy",
            system="vanilla_equation", law_version="v0", **inp)
        pt = recover_force("vanilla_equation", inp, raw)
        if pt is not None:
            row = dict(pt.inputs); row["force"] = pt.force
            data_points.append(row)

    adapter = NewtonAdapter(data_points, var_names=params, target_name="force")
    res = engine.run(adapter)
    best = res.winners[0] if res.winners else None
    loss = best[1].loss_mean if best else 1.0
    return {
        "benchmark": "newtonbench", "metric": "1-loss", "score": round(1 - loss, 4), "max": 1,
        "normalized": max(0.0, 1 - loss),
        "detail": {"best_law": best[0].code if best else None,
                   "const": getattr(best[0], "_const", None) if best else None,
                   "loss": loss, "n_valid": res.n_valid},
        "supermetrics": res.supermetrics,
    }


def run_uh_bio(engine: UniversalCPI, seed: int, judge_model: str) -> dict[str, Any]:
    from envs.bio_env import env as bem
    bem.GeneticsLabEnvironment.load_judge_config = lambda self: _judge_cfg(judge_model)
    from envs.bio_env.env import GeneticsLabEnvironment
    from envs.common import Difficulty

    with contextlib.redirect_stdout(io.StringIO()):
        env = GeneticsLabEnvironment(seed=seed, difficulty=Difficulty.HARD, required_steps=18)

    cross_records = []
    async def collect():
        q = await env.query_organisms(1, 10)
        ids = [o["id"] for o in q.get("organisms", [])][:3]
        if len(ids) < 3:
            ids = [1, 2, 3]
        pairs = [(ids[0],ids[1]),(ids[0],ids[2]),(ids[1],ids[2])] * 6
        for p1, p2 in pairs:
            if env.current_experiments >= 18:
                break
            if len(env.organisms) > 150:
                extra = [oid for oid in list(env.organisms.keys()) if oid > 10]
                with contextlib.redirect_stdout(io.StringIO()):
                    await env.remove_organisms(extra)
            with contextlib.redirect_stdout(io.StringIO()):
                r = await env.conduct_cross(p1, p2, 12)
            if r.get("success"):
                cross_records.append(r)
    _run_async(collect())

    adapter = BioAdapter(cross_records, judge_model=judge_model)
    res = engine.run(adapter)

    async def judge():
        with contextlib.redirect_stdout(io.StringIO()):
            return await env.commit_final_result(res.report)
    commit = _run_async(judge())
    score = 0.0
    if commit.get("success"):
        jr = commit.get("judge_result", commit.get("result", {}).get("judge_result", {}))
        score = float(jr.get("final_score", 0.0))
    return {
        "benchmark": "uh_bio", "metric": "official_judge_100", "score": score, "max": 100,
        "normalized": score / 100,
        "detail": {"n_crosses": len(cross_records), "n_valid": res.n_valid,
                   "winners": [w[0].name for w in res.winners[:3]]},
        "supermetrics": res.supermetrics,
    }


def run_discoverybench(engine: UniversalCPI, seed: int, judge_model: str) -> dict[str, Any]:
    import pandas as pd
    from mars.runners.run_db_official_eval import load_official_real_tasks, _run_official_eval

    task_key = "worldbank_education_gdp:3:0"
    dataset = "worldbank_education_gdp"
    meta_id = 3
    task_dir = _PROJ / "discoverybench_repo" / "discoverybench" / "real" / "test" / dataset
    meta = json.load(open(task_dir / f"metadata_{meta_id}.json"))
    question = meta["queries"][0][0]["question"]
    dk = meta.get("domain_knowledge", "")[:600]
    ds = meta["datasets"][0]
    df = pd.read_csv(task_dir / ds["name"])
    for col in df.columns:
        if df[col].dtype == object:
            try:
                df[col] = df[col].str.replace(",", ".", regex=False).astype(float)
            except Exception:
                pass
    col_descs = {c["name"]: c["description"] for c in ds.get("columns", {}).get("raw", []) if c.get("name")}

    adapter = DiscoveryAdapter(df, question, dk, col_descs, judge_model=judge_model)
    res = engine.run(adapter)

    tasks = load_official_real_tasks(_PROJ / "discoverybench_repo", max_tasks=None,
        datasets={dataset}, task_keys={task_key})
    hms = 0.0
    if tasks:
        ev = _run_official_eval(tasks[0], pred_hypo=res.report, pred_workflow="",
            judge_model=judge_model, trace_f=io.StringIO())
        hms = 100.0 * float(ev.get("final_score", 0.0))
    return {
        "benchmark": "discoverybench", "metric": "official_HMS_100", "score": round(hms, 1), "max": 100,
        "normalized": hms / 100,
        "detail": {"task": task_key, "n_valid": res.n_valid, "report": res.report[:200]},
        "supermetrics": res.supermetrics,
    }


DRIVERS = {
    "uh_seq": run_uh_seq,
    "newtonbench": run_newtonbench,
    "uh_bio": run_uh_bio,
    "discoverybench": run_discoverybench,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", default="universal_suite_smoke")
    parser.add_argument("--benchmarks", default="uh_seq,newtonbench,uh_bio,discoverybench")
    parser.add_argument("--model", default="openai/gpt-4o")
    parser.add_argument("--judge_model", default="openai/gpt-4o")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_proposals", type=int, default=14)
    parser.add_argument("--max_rounds", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = _PROJ / "lmw" / "universal_cpi" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    cpi_log_path = out_dir / "cpi_results.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"exists; pass --overwrite: {out_path}")

    # ONE engine for all benchmarks
    engine = UniversalCPI(model=args.model, n_proposals=args.n_proposals, max_rounds=args.max_rounds)

    benches = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    results = []
    print(f"=== Universal CPI Suite: ONE engine on {len(benches)} benchmarks ===")
    print(f"model={args.model}  seed={args.seed}\n")

    for b in benches:
        if b not in DRIVERS:
            print(f"  ! unknown benchmark: {b}")
            continue
        print(f"  running {b} ...", flush=True)
        t0 = time.time()
        log_start = len(engine.result_log)
        try:
            row = DRIVERS[b](engine, args.seed, args.judge_model)
            row["wall_time_s"] = round(time.time() - t0, 1)
        except Exception as e:
            import traceback
            row = {"benchmark": b, "error": str(e), "traceback": traceback.format_exc()[:500],
                   "normalized": 0.0, "wall_time_s": round(time.time() - t0, 1)}
        log_end = len(engine.result_log)
        row["cpi_result_range"] = [log_start, log_end]
        results.append(row)
        sc = row.get("score", "?")
        mx = row.get("max", "?")
        print(f"    {b}: {sc}/{mx}  (normalized {row.get('normalized',0):.2f})  {row.get('wall_time_s')}s")

    mean_norm = sum(r.get("normalized", 0) for r in results) / len(results) if results else 0
    mean_super = sum(_supermetric_score(r) for r in results) / len(results) if results else 0
    cpi_supermetrics = [
        row.get("supermetrics", {})
        for row in engine.result_log
        if isinstance(row, dict) and isinstance(row.get("supermetrics"), dict)
    ]
    aggregate_internal = aggregate_supermetrics(cpi_supermetrics)
    summary = {
        "run_id": args.run_id,
        "score_type": "universal-autonomous-engine",
        "claim": "ONE UniversalCPI engine + thin adapters (no domain answers) across 4 benchmarks",
        "model": args.model,
        "judge_model": args.judge_model,
        "seed": args.seed,
        "n_benchmarks": len(results),
        "mean_normalized": round(mean_norm, 4),
        "mean_supermetric_universal_score": round(mean_super, 4),
        "aggregate_internal_supermetrics": aggregate_internal,
        "results": results,
    }
    out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    cpi_log_path.write_text(
        json.dumps(
            {
                "run_id": args.run_id,
                "model": args.model,
                "judge_model": args.judge_model,
                "results": engine.result_log,
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    print(f"\n=== SUMMARY (one engine, autonomous) ===")
    print(f"{'Benchmark':<16} {'Score':<14} {'Normalized':<10}")
    print("-" * 42)
    for r in results:
        print(f"{r['benchmark']:<16} {str(r.get('score','err'))+'/'+str(r.get('max','?')):<14} {r.get('normalized',0):.2f}")
    print("-" * 42)
    print(f"{'MEAN':<16} {'':<14} {mean_norm:.2f}")
    print(f"Mean supermetric universal_score: {mean_super:.2f}")
    epistemic = aggregate_internal.get("epistemic", {})
    if epistemic:
        print(
            "Epistemic certificates: "
            f"accepted={epistemic.get('accepted', 0)} "
            f"provisional={epistemic.get('provisional', 0)} "
            f"rejected={epistemic.get('rejected', 0)} "
            f"mean_leakage={epistemic.get('mean_leakage_penalty', 0.0)}"
        )
    print(f"\nsummary → {out_path}")
    print(f"cpi log → {cpi_log_path}")


if __name__ == "__main__":
    main()
