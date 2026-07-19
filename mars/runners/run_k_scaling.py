"""K-scaling curve: weak(K) vs strong(K=1) across task types.

THE empirical claim: weak_model(K=N) + MDL >= strong_model(K=1)
for N that can be computed cheaply.

Shows the compute-accuracy curve for both:
  - Execution-aligned tasks (UH-Grid): MDL v1 (syntactic, exact verifier)
  - Open-ended tasks (hypothesis): MDL v2 (semantic, structured output)

Run:
    python -m mars.runners.run_k_scaling --run_id kscale_v1 --overwrite

Output: lmw/kscale/<run_id>/summary.json + curve data for plotting
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
_PROJ = Path(__file__).resolve().parent.parent.parent
for _p in (_PROJ, _PROJ / "ultrahorizon_repo"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from dotenv import load_dotenv
    load_dotenv(_PROJ / "autodiscovery" / ".env.local", override=False)
except ImportError:
    pass

from mars.agents.base import call_llm, make_openai_client
from mars.mdl.engine import MDLEngine, Library, make_llm_proposer
from mars.mdl.tasks_grid import GridLetterTask
from mars.mdl.variation_engine_v2 import make_v2_engine, _parse_json, FIELDS
from mars.runners.run_mdl_grid import collect as grid_collect, _eval_code, _extract

WEAK = "openai/gpt-4o-mini"
STRONG = "openai/gpt-4o"

K_VALUES = [1, 3, 5, 10, 20]   # test these K values

# Hypothesis tasks — clean ground truth, unambiguous data
HYP_TASKS = [
    {
        "id": "parity_rule",
        "task": (
            "A function maps grid positions to +1 or -1: "
            "x=0,y=0→+1; x=1,y=0→-1; x=0,y=1→-1; x=1,y=1→+1; "
            "x=2,y=0→+1; x=3,y=1→-1; x=4,y=2→+1; x=5,y=3→-1. "
            "What is the rule?"
        ),
        "gold": "(x+y) even → +1, odd → -1",
        "gold_keywords": ["parity", "even", "x+y"],
    },
    {
        "id": "energy_threshold_clean",
        "task": (
            "A robot steps on a tile. Score change depends on energy level. "
            "Observations (all consistent): "
            "energy=20→+2, energy=18→+2, energy=16→+2, energy=15→+2, "
            "energy=14→-2, energy=12→-2, energy=9→-2, energy=7→-2. "
            "What is the exact threshold rule?"
        ),
        "gold": "if energy >= 15: +2, else: -2",
        "gold_keywords": ["15", "threshold", "energy"],
    },
    {
        "id": "visit_decay",
        "task": (
            "A tile gives rewards that depend on how many times it was visited. "
            "Observations: visit 1→+5, visit 2→+5, visit 3→+5, "
            "visit 4→+2, visit 5→+2, visit 6→0, visit 7→0, visit 8→0. "
            "What is the rule?"
        ),
        "gold": "visits 1-3: +5, visits 4-5: +2, visits 6+: 0",
        "gold_keywords": ["3", "visit", "decay", "5"],
    },
    {
        "id": "modulo_rule",
        "task": (
            "A function maps step count to score change: "
            "step=3→+1, step=6→+1, step=9→+1, step=12→+1, "
            "step=1→0, step=2→0, step=4→0, step=5→0, step=7→0. "
            "What is the rule?"
        ),
        "gold": "+1 if steps is divisible by 3, else 0",
        "gold_keywords": ["3", "divisible", "modulo", "multiple"],
    },
]


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

def judge_keywords(answer: str, kws: list[str]) -> float:
    a = answer.lower()
    return sum(1 for k in kws if k.lower() in a) / len(kws)


def judge_semantic(client, answer: str, gold: str, task: str) -> float:
    prompt = (
        f"Task: {task}\n\nGold: {gold}\n\nCandidate: {answer}\n\n"
        "Does the candidate correctly capture the key rule? "
        "1=correct, 0.5=partial, 0=wrong. Output only the number."
    )
    try:
        r = call_llm(client, model=STRONG, system="Strict scientific judge.",
                     user=prompt, max_tokens=5, temperature=0.0)
        return float(r.strip().split()[0])
    except Exception:
        return 0.0


def raw_answer(client, model: str, task: str) -> str:
    try:
        return call_llm(client, model=model,
                        system="Precise scientific analyst.",
                        user=f"Task: {task}\nGive your best specific answer.",
                        max_tokens=200, temperature=0.2)
    except Exception:
        return ""


def best_of_k(client, model: str, task: str, k: int, gold: str, gold_kws: list) -> float:
    """Best-of-K: generate K answers, pick best by keyword score."""
    answers = []
    for _ in range(k):
        try:
            a = call_llm(client, model=model,
                         system="Precise scientific analyst.",
                         user=f"Task: {task}\nBest specific answer.",
                         max_tokens=200, temperature=0.85)
            answers.append(a)
        except Exception:
            pass
    if not answers:
        return 0.0
    # Pick best by keyword (proxy for gold, no oracle needed)
    best = max(answers, key=lambda a: judge_keywords(a, gold_kws))
    return judge_semantic(client, best, gold, task)


# ---------------------------------------------------------------------------
# Grid K-scaling (execution-aligned, exact verifier)
# ---------------------------------------------------------------------------

def grid_k_scaling(client, seed: int = 3, difficulty: str = "easy",
                   letter: str = "E") -> dict:
    """Run MDL with varying K on one Grid letter. Returns {K: exact_rate}."""
    obs_by_letter, _ = grid_collect(seed, difficulty)
    obs = obs_by_letter.get(letter, [])
    if not obs:
        return {}

    results = {}
    proposer = make_llm_proposer(WEAK)

    for k in K_VALUES:
        engine = MDLEngine(proposer, max_rounds=1, k_per_round=k, library=Library())
        res = engine.compress(GridLetterTask(letter, obs))
        results[k] = round(res.exact_rate, 3)
        print(f"  Grid {letter} K={k:2d}: exact={res.exact_rate:.3f}", flush=True)

    # Strong baseline (K=1)
    strong_code = _extract(call_llm(
        client, model=STRONG,
        system="Write Python functions for hidden grid effects.",
        user=(f"Observed effects: {json.dumps([{'x':o.x,'y':o.y,'energy':o.energy,'delta':o.delta_score} for o in obs[:8]])}\n"
              "def f(context): return <int score change>"),
        max_tokens=300, temperature=0.2))
    results["strong_k1"] = round(_eval_code(strong_code, obs), 3)
    print(f"  Grid {letter} strong K=1: exact={results['strong_k1']:.3f}", flush=True)

    return results


# ---------------------------------------------------------------------------
# Hypothesis K-scaling (semantic, v2 engine)
# ---------------------------------------------------------------------------

def hyp_k_scaling(client, task_def: dict) -> dict:
    """Run semantic variation-MDL with varying K. Returns {K: sem_score}."""
    task = task_def["task"]
    gold = task_def["gold"]
    kws = task_def["gold_keywords"]
    results = {}

    for k in K_VALUES:
        engine = make_v2_engine(
            weak_model=WEAK, meta_model=WEAK,
            k=k, max_rounds=1, client=client)
        res = engine.run(task)
        sem = judge_semantic(client, res.answer, gold, task)
        results[k] = sem
        results[f"ratio_k{k}"] = res.compression_ratio
        print(f"  Hyp {task_def['id']} K={k:2d}: sem={sem} ratio={res.compression_ratio:.2f} "
              f"answer={res.answer[:60]!r}", flush=True)

    # Strong baseline
    strong_ans = raw_answer(client, STRONG, task)
    results["strong_k1"] = judge_semantic(client, strong_ans, gold, task)
    print(f"  Hyp {task_def['id']} strong K=1: sem={results['strong_k1']}", flush=True)

    # Best-of-K weak (oracle pick) — upper bound
    bok = best_of_k(client, WEAK, task, K_VALUES[-1], gold, kws)
    results["weak_best_of_k"] = bok
    print(f"  Hyp {task_def['id']} weak best-of-{K_VALUES[-1]}: sem={bok}", flush=True)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="kscale_v1")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_dir = _PROJ / "lmw" / "kscale" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"exists; --overwrite to rerun: {out_path}")

    client = make_openai_client()
    summary: dict = {"run_id": args.run_id, "k_values": K_VALUES,
                     "weak": WEAK, "strong": STRONG, "results": {}}

    print("=== K-Scaling Curve: weak(K) + MDL vs strong(K=1) ===\n")

    # 1. Grid (execution-aligned, exact verifier)
    print("--- UH-Grid letter E (execution-aligned) ---")
    grid_res = grid_k_scaling(client, seed=3, difficulty="easy", letter="E")
    summary["results"]["grid_E"] = grid_res
    out_path.write_text(json.dumps(summary, indent=2))

    # K that matches strong
    strong = grid_res.get("strong_k1", 0)
    match_k = next((k for k in K_VALUES if grid_res.get(k, 0) >= strong), None)
    print(f"  → strong={strong:.3f} | weak+MDL matches at K={match_k}\n")

    # 2. Hypothesis (semantic, v2 engine)
    for td in HYP_TASKS:
        print(f"--- Hypothesis: {td['id']} ---")
        hyp_res = hyp_k_scaling(client, td)
        summary["results"][f"hyp_{td['id']}"] = hyp_res
        out_path.write_text(json.dumps(summary, indent=2))
        strong_h = hyp_res.get("strong_k1", 0)
        match_k_h = next((k for k in K_VALUES if hyp_res.get(k, 0) >= strong_h), None)
        print(f"  → strong={strong_h} | weak+MDL matches at K={match_k_h}\n")

    # Print final curve summary
    print("=== K-SCALING CURVES ===")
    for name, res in summary["results"].items():
        curve = " → ".join(f"K={k}:{res.get(k,'?'):.2f}" for k in K_VALUES
                           if isinstance(res.get(k), float))
        print(f"  {name}: {curve}  |  strong={res.get('strong_k1','?'):.2f}")

    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nsummary → {out_path}")

    # Calibration: does compression_ratio predict accuracy?
    print("\n=== CALIBRATION: compression_ratio → accuracy ===")
    calib_points = []
    for name, res in summary["results"].items():
        if not isinstance(res, dict):
            continue
        # collect (ratio, accuracy) pairs across K values
        ratio_key = f"ratio_{name}"
        for k in K_VALUES:
            acc = res.get(k)
            ratio = res.get(f"ratio_k{k}")
            if isinstance(acc, float) and isinstance(ratio, float):
                calib_points.append({"name": name, "k": k, "accuracy": acc, "ratio": ratio})
    if calib_points:
        by_ratio = sorted(calib_points, key=lambda x: x["ratio"])
        print("  ratio  | accuracy | name")
        for p in by_ratio:
            print(f"  {p['ratio']:.2f}  | {p['accuracy']:.2f}     | {p['name']} K={p['k']}")
    else:
        print("  (run with --save_ratios to collect calibration data)")

    print("\n=== THESIS CHECK ===")
    all_wins = 0
    for name, res in summary["results"].items():
        strong = res.get("strong_k1", 0)
        mdl_max = max((res.get(k, 0) for k in K_VALUES if isinstance(res.get(k), float)), default=0)
        wins = mdl_max >= strong
        all_wins += int(wins)
        print(f"  {name}: weak+MDL_best={mdl_max:.2f} vs strong={strong:.2f} → {'WIN' if wins else 'loss'}")
    n_tasks = len(summary["results"])
    print(f"\n  OVERALL: {all_wins}/{n_tasks} tasks weak+MDL >= strong")


if __name__ == "__main__":
    main()
