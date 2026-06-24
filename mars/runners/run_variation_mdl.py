"""Variation-MDL scaling experiment.

Shows the K-scaling curve: does weak(K=20) > strong(K=1)?
Tests on open-ended hypothesis generation — NO verifier needed.

Uses DiscoveryBench-style tasks: given a dataset description + findings,
generate the correct hypothesis. Judge = gpt-4o semantic match (0/1).

Run:
    python -m mars.runners.run_variation_mdl --run_id varmdl_v1 --k 20 --rounds 3
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
    load_dotenv(_PROJ / "autodiscovery" / ".env.local", override=True)
except ImportError:
    pass

from mars.agents.base import call_llm, make_openai_client
from mars.mdl.variation_engine import make_variation_engine, _bits, _mdl_score, _baseline

WEAK = "openai/gpt-4o-mini"
STRONG = "openai/gpt-4o"

# ---------------------------------------------------------------------------
# Toy hypothesis tasks (no external dataset needed for first test)
# Each task has: a description of observations, the ground truth hypothesis
# ---------------------------------------------------------------------------

TASKS = [
    {
        "id": "energy_threshold",
        "task": (
            "A robot in a 10×10 grid collects observations when stepping on tile E. "
            "energy=19→score+2, energy=13→score-2, energy=11→score-2, energy=20→score+2, "
            "energy=8→score-2, energy=17→score+2, energy=14→score-1. "
            "What is the rule that determines the score change?"
        ),
        "gold": "if energy >= 15: +2, else: -2",
        "gold_keywords": ["15", "energy", "threshold"],
    },
    {
        "id": "parity_rule",
        "task": (
            "A function maps grid positions to effects. "
            "x=0,y=0→+1; x=1,y=0→-1; x=0,y=1→-1; x=1,y=1→+1; "
            "x=2,y=0→+1; x=3,y=1→-1; x=4,y=2→+1. "
            "What is the rule?"
        ),
        "gold": "if (x+y) is even: +1, else: -1",
        "gold_keywords": ["parity", "even", "x+y", "sum"],
    },
    {
        "id": "visit_count",
        "task": (
            "A tile gives score changes based on visit count: "
            "visit 1→+3, visit 2→+3, visit 3→+3, visit 4→+1, visit 5→+1, visit 6→0. "
            "What is the rule for the score change?"
        ),
        "gold": "if visit_count <= 3: +3, elif visit_count <= 5: +1, else: 0",
        "gold_keywords": ["visit", "3", "count", "decreasing"],
    },
    {
        "id": "gene_interaction",
        "task": (
            "In a genetics experiment: cross AA×BB → 75% large offspring; "
            "cross AA×bb → 50% large; cross aa×BB → 50% large; cross aa×bb → 25% large. "
            "What is the genetic rule?"
        ),
        "gold": "each dominant allele (A or B) independently increases probability of large by 25%",
        "gold_keywords": ["dominant", "additive", "independent", "probability"],
    },
    {
        "id": "chemical_reaction",
        "task": (
            "Reaction rate data: T=300K→rate=0.01, T=350K→rate=0.08, "
            "T=400K→rate=0.5, T=450K→rate=2.8. "
            "What is the mathematical relationship between temperature and rate?"
        ),
        "gold": "Arrhenius: rate = A * exp(-Ea/RT), exponential in 1/T",
        "gold_keywords": ["exponential", "arrhenius", "temperature", "activation"],
    },
]


# ---------------------------------------------------------------------------
# Judge: does the answer contain the key concepts?
# ---------------------------------------------------------------------------

def judge_keywords(answer: str, gold_keywords: list[str]) -> float:
    """Simple keyword-based judge (no API call, fast)."""
    answer_lower = answer.lower()
    hits = sum(1 for kw in gold_keywords if kw.lower() in answer_lower)
    return hits / len(gold_keywords)


def judge_semantic(client, answer: str, gold: str, task: str) -> float:
    """LLM judge: does this answer capture the correct rule?"""
    prompt = (
        f"Task: {task}\n\n"
        f"Gold answer: {gold}\n\n"
        f"Candidate answer: {answer}\n\n"
        "Does the candidate answer correctly capture the key rule or relationship? "
        "Be strict: it must get the direction, threshold, or mechanism right. "
        "Answer: 1 if correct, 0 if wrong, 0.5 if partially correct. Output only the number."
    )
    try:
        r = call_llm(client, model=STRONG,
                     system="You are a strict scientific judge.",
                     user=prompt, max_tokens=10, temperature=0.0)
        return float(r.strip().split()[0])
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def raw_answer(client, model: str, task: str) -> str:
    """Single-shot answer."""
    try:
        return call_llm(client, model=model,
                        system="You are a precise scientific analyst.",
                        user=f"Task: {task}\n\nGive your best answer. Be specific.",
                        max_tokens=300, temperature=0.2)
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="varmdl_v1")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_dir = _PROJ / "lmw" / "varmdl" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"exists; pass --overwrite: {out_path}")

    client = make_openai_client()
    engine = make_variation_engine(
        weak_model=WEAK, meta_model=WEAK,
        k=args.k, max_rounds=args.rounds, client=client)

    rows = []
    print(f"=== Variation-MDL: weak×K vs strong×1 (no verifier needed) ===")
    print(f"weak={WEAK} strong={STRONG} K={args.k} rounds={args.rounds}\n")

    for task_def in TASKS:
        tid = task_def["id"]
        task_str = task_def["task"]
        gold = task_def["gold"]
        kws = task_def["gold_keywords"]

        print(f"[{tid}]", flush=True)

        # Baselines
        weak_ans = raw_answer(client, WEAK, task_str)
        strong_ans = raw_answer(client, STRONG, task_str)

        weak_kw = judge_keywords(weak_ans, kws)
        strong_kw = judge_keywords(strong_ans, kws)

        # Variation-MDL (weak × K)
        t0 = time.time()
        result = engine.run(task_str)
        elapsed = time.time() - t0

        mdl_kw = judge_keywords(result.core, kws)

        # Semantic judge on all three
        weak_sem = judge_semantic(client, weak_ans, gold, task_str)
        strong_sem = judge_semantic(client, strong_ans, gold, task_str)
        mdl_sem = judge_semantic(client, result.core, gold, task_str)

        row = {
            "task_id": tid,
            "weak_raw_kw": round(weak_kw, 2),
            "strong_raw_kw": round(strong_kw, 2),
            "weak_mdl_kw": round(mdl_kw, 2),
            "weak_raw_sem": weak_sem,
            "strong_raw_sem": strong_sem,
            "weak_mdl_sem": mdl_sem,
            "compression_ratio": result.compression_ratio,
            "uncertainty": result.uncertainty,
            "n_responses": result.n_responses,
            "rounds": result.rounds,
            "probe_questions": result.probe_questions,
            "core_preview": result.core[:200],
            "elapsed_s": round(elapsed, 1),
        }
        rows.append(row)

        print(f"  weak_raw  kw={weak_kw:.2f} sem={weak_sem}")
        print(f"  strong_raw kw={strong_kw:.2f} sem={strong_sem}")
        print(f"  weak+MDL  kw={mdl_kw:.2f} sem={mdl_sem}  "
              f"ratio={result.compression_ratio:.2f} n={result.n_responses}")
        print(f"  core: {result.core[:100]!r}")
        if result.probe_questions:
            print(f"  probes: {result.probe_questions[:2]}")
        print()

        out_path.write_text(json.dumps({"partial": rows}, indent=2))

    # Summary
    n = len(rows)
    def mean(key): return round(sum(r[key] for r in rows) / n, 3) if n else 0

    summary = {
        "run_id": args.run_id, "k": args.k, "rounds": args.rounds,
        "weak": WEAK, "strong": STRONG, "n_tasks": n,
        "mean_weak_raw_sem": mean("weak_raw_sem"),
        "mean_strong_raw_sem": mean("strong_raw_sem"),
        "mean_weak_mdl_sem": mean("weak_mdl_sem"),
        "mean_compression_ratio": mean("compression_ratio"),
        "thesis_holds_sem": mean("weak_mdl_sem") >= mean("strong_raw_sem"),
        "results": rows,
    }
    out_path.write_text(json.dumps(summary, indent=2))

    print("=== MEAN (semantic judge) ===")
    print(f"  weak_raw  {mean('weak_raw_sem'):.3f}")
    print(f"  strong_raw {mean('strong_raw_sem'):.3f}")
    print(f"  weak+MDL  {mean('weak_mdl_sem'):.3f}  "
          f"{'THESIS HOLDS' if summary['thesis_holds_sem'] else '< strong'}")
    print(f"  mean compression ratio: {mean('compression_ratio'):.2f}")
    print(f"\nsummary -> {out_path}")


if __name__ == "__main__":
    main()
