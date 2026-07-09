"""Official NewtonBench through UniversalCPI + DPSR — directly comparable SA.

Bridges the authors' official environment (modules.*.run_experiment_for_module /
evaluate_law) to our universal engine. Observations are collected from the real
env; the winning law is scored by the authors' own evaluate_law (exact_accuracy =
the SA metric in the paper). DPSR participates as a pre-proposal layer.

Published SA baselines (avg over 324 tasks): GPT-5 75.9, Gemini-2.5-pro 65.4,
o4-mini 47.8, DeepSeek-R1 43.4.

Run (small slice first):
  python -m mars.runners.run_nb_official_dpsr --run_id nb_off_v1 \
    --modules m0_gravity,m9_hooke_law --difficulty easy --overwrite
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")
_PROJ = Path(__file__).resolve().parent.parent.parent
_NB = _PROJ / "newtonbench_repo"
for _p in (_PROJ, _NB):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
try:
    from dotenv import load_dotenv
    load_dotenv(_PROJ / "autodiscovery" / ".env.local", override=True)
except ImportError:
    pass

from mars.induction.cpi_adapters import NewtonAdapter
from mars.induction.universal_cpi import UniversalCPI

WEAK = "openai/gpt-4o-mini"


def _params(signature: str) -> list[str]:
    inside = signature[signature.index("(") + 1: signature.rindex(")")]
    return [p.split(":")[0].split("=")[0].strip() for p in inside.split(",") if p.strip()]


def collect_official(module, params, *, difficulty, law_version, system,
                     n: int, seed: int) -> list[dict]:
    """Sample inputs, query the real env, return data points {param:..., y:force}."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        inp = {p: round(rng.uniform(0.5, 5.0), 3) for p in params}
        try:
            raw = module.run_experiment_for_module(
                noise_level=0.0, difficulty=difficulty, system=system,
                law_version=law_version, **inp)
            y = float(raw)
            if y == y:  # not NaN
                rows.append({**inp, "y": y})
        except Exception:
            continue
    return rows


def submit_code(signature: str, params: list[str], winner_code: str, k: float) -> str:
    """Wrap the engine's law(inputs) winner into the module's official signature."""
    dict_lit = "{" + ", ".join(f"{p!r}: {p}" for p in params) + "}"
    return (
        f"{winner_code}\n\n"
        f"{signature}\n"
        f"    return {k!r} * law({dict_lit})\n"
    )


def run_task(module_name, *, difficulty, law_version, system, dpsr_on, seed) -> dict:
    os.environ["MARS_DPSR"] = "1" if dpsr_on else "0"
    module = importlib.import_module(f"modules.{module_name}")
    signature = str(module.FUNCTION_SIGNATURE).strip()
    params = _params(signature)
    rows = collect_official(module, params, difficulty=difficulty,
                            law_version=law_version, system=system, n=24, seed=seed)
    if len(rows) < 6:
        return {"SA": 0.0, "error": "too few observations"}

    adapter = NewtonAdapter(rows, var_names=params, target_name="y")
    engine = UniversalCPI(model=WEAK, n_proposals=10, max_rounds=2, holdout_frac=0.4)
    result = engine.run(adapter)
    if not result.winners:
        return {"SA": 0.0, "error": "no winner"}

    best, score = result.winners[0]
    k = float(getattr(best, "_const", 1.0))
    law_code = submit_code(signature, params, best.code, k)
    try:
        ev = module.evaluate_law(
            law_code, param_description=getattr(module, "PARAM_DESCRIPTION", ""),
            difficulty=difficulty, law_version=law_version, judge_model_name="gpt41")
    except Exception as e:
        return {"SA": 0.0, "error": f"evaluate_law: {type(e).__name__}: {e}",
                "submitted": law_code[:300]}
    import math as _m
    rmsle = ev.get("rmsle", float("nan"))
    num_acc = float(_m.exp(-rmsle)) if rmsle == rmsle else 0.0
    return {
        "SA": float(ev.get("exact_accuracy", 0.0)),
        "numeric_accuracy": round(num_acc, 4),
        "rmsle": rmsle,
        "symbolic_equivalent": bool(ev.get("symbolic_equivalent", False)),
        "holdout_loss": round(score.loss_mean, 4),
        "winner_tags": list(best.tags),
        "submitted": law_code[:400],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="nb_off_v1")
    ap.add_argument("--modules", default="m0_gravity,m9_hooke_law")
    ap.add_argument("--difficulty", default="easy")
    ap.add_argument("--law_version", default="v0")
    ap.add_argument("--system", default="vanilla_equation")
    ap.add_argument("--ablation", action="store_true", help="also run DPSR-off baseline")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    out_dir = _PROJ / "lmw" / "nb_official" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"exists; --overwrite: {out_path}")

    modules = [m.strip() for m in args.modules.split(",") if m.strip()]
    summary: dict[str, Any] = {"run_id": args.run_id, "weak": WEAK,
                               "difficulty": args.difficulty, "results": {}}
    print(f"=== Official NewtonBench via UniversalCPI+DPSR ({args.difficulty}) ===\n")
    t0 = time.time()
    sa_on, sa_off, num_on = [], [], []
    for mod in modules:
        on = run_task(mod, difficulty=args.difficulty, law_version=args.law_version,
                      system=args.system, dpsr_on=True, seed=args.seed)
        sa_on.append(on.get("SA", 0.0))
        num_on.append(on.get("numeric_accuracy", 0.0))
        row = {"dpsr_on": on}
        line = (f"  {mod:18} DPSR-on  SA={on.get('SA',0.0):.3f} "
                f"num={on.get('numeric_accuracy',0.0):.3f}  {on.get('winner_tags','')}")
        if on.get("error"):
            line += f"  [{on['error'][:60]}]"
        print(line, flush=True)
        if args.ablation:
            off = run_task(mod, difficulty=args.difficulty, law_version=args.law_version,
                           system=args.system, dpsr_on=False, seed=args.seed)
            sa_off.append(off.get("SA", 0.0))
            row["dpsr_off"] = off
            print(f"  {mod:18} DPSR-off SA={off.get('SA',0.0):.3f}", flush=True)
        summary["results"][mod] = row
        out_path.write_text(json.dumps(summary, indent=2))

    n = len(modules)
    mean_on = sum(sa_on) / n if n else 0.0
    mean_num = sum(num_on) / n if n else 0.0
    print(f"\n=== SUMMARY ({n} tasks, {args.difficulty}) ===")
    print(f"  weak+CPI+DPSR  mean SA = {mean_on:.3f}   mean numeric_accuracy = {mean_num:.3f}")
    summary["mean_numeric_accuracy"] = round(mean_num, 4)
    if args.ablation and sa_off:
        print(f"  weak+CPI (no DPSR) mean SA = {sum(sa_off)/n:.3f}")
    print(f"  published: GPT-5 0.759, Gemini-2.5-pro 0.654, o4-mini 0.478, DeepSeek-R1 0.434")
    summary["mean_SA_dpsr_on"] = round(mean_on, 4)
    if sa_off:
        summary["mean_SA_dpsr_off"] = round(sum(sa_off) / n, 4)
    summary["wall_time_s"] = round(time.time() - t0, 1)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nsummary → {out_path}")


if __name__ == "__main__":
    main()
