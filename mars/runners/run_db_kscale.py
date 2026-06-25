"""DiscoveryBench K-scaling: weak(K)+MDL vs strong(K=1).

Uses DiscoveryBench-Synth tasks (gold hypothesis available).
Context = column descriptions + first 15 rows + basic stats.
No code execution needed — model infers from context.

Run:
    python -m mars.runners.run_db_kscale --run_id db_kscale_v1
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import pandas as pd

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
from mars.mdl.variation_engine_v2 import make_v2_engine

WEAK = "openai/gpt-4o-mini"
STRONG = "openai/gpt-4o"
K_VALUES = [1, 3, 5, 10]


# ---------------------------------------------------------------------------
# Task loader
# ---------------------------------------------------------------------------

def load_db_tasks(n: int = 16) -> list[dict]:
    """Load up to n diverse synth tasks with qualitative gold hypotheses."""
    import glob

    tasks = []
    for mf in sorted(glob.glob(
        str(_PROJ / "discoverybench_repo/discoverybench/synth/train/*/metadata_0.json")
    )):
        try:
            m = json.load(open(mf, encoding="utf-8", errors="replace"))
        except Exception:
            continue
        q = m.get("queries", [])
        if not q:
            continue
        q0 = q[0] if isinstance(q[0], dict) else (q[0][0] if isinstance(q[0], list) and q[0] else None)
        if not q0:
            continue
        gold = q0.get("true_hypothesis", "")
        if not gold or len(gold) > 250:
            continue
        # Skip tasks requiring exact formulas
        skip_words = ["computed as", "= 3 +", "^2", "sqrt", "log(", "coefficient"]
        if any(w in gold for w in skip_words):
            continue
        # Prefer directional hypotheses
        dir_words = ["higher", "lower", "more", "less", "positive", "negative",
                     "greater", "increases", "decreases", "tend to"]
        if not any(w in gold.lower() for w in dir_words):
            continue
        domain = m.get("domain", "")
        datasets = m.get("datasets", [])
        if not datasets:
            continue
        ds = datasets[0]
        csv_name = ds.get("name", "data.csv")
        csv_path = Path(mf).parent / csv_name
        if not csv_path.exists():
            continue
        cols = ds.get("columns", [])
        tasks.append({
            "name": Path(mf).parent.name,
            "domain": domain,
            "question": q0.get("question", ""),
            "gold": gold,
            "csv": str(csv_path),
            "cols": cols,
        })

    # Pick up to 2 per domain, up to n total
    by_domain: dict[str, list] = defaultdict(list)
    for t in tasks:
        by_domain[t["domain"]].append(t)
    selected = []
    for ts in sorted(by_domain.values(), key=lambda x: x[0]["domain"]):
        selected.extend(ts[:2])
        if len(selected) >= n:
            break
    return selected[:n]


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def build_context(task: dict) -> str:
    """Column descriptions + first 15 rows + numeric correlations."""
    cols = task["cols"]
    col_desc = "\n".join(
        f"  - {c['name']}: {c.get('description', '')}" for c in cols[:20]
    )

    try:
        df = pd.read_csv(task["csv"])
        sample = df.head(15).to_string(index=False, max_cols=10)
        # basic numeric correlations (top pairs)
        num = df.select_dtypes("number")
        corr_lines = []
        if num.shape[1] >= 2:
            corr = num.corr()
            pairs = []
            for i, c1 in enumerate(corr.columns):
                for c2 in corr.columns[i+1:]:
                    r = corr.loc[c1, c2]
                    if abs(r) > 0.2:
                        pairs.append((abs(r), r, c1, c2))
            for _, r, c1, c2 in sorted(pairs, reverse=True)[:5]:
                corr_lines.append(f"  {c1} vs {c2}: r={r:.2f}")
        corr_text = "\n".join(corr_lines) if corr_lines else "  (no strong correlations)"
    except Exception as e:
        sample = f"(error reading CSV: {e})"
        corr_text = ""

    return (
        f"Dataset columns:\n{col_desc}\n\n"
        f"Sample rows (first 15):\n{sample}\n\n"
        f"Numeric correlations (|r|>0.2):\n{corr_text}\n\n"
        f"Discovery query: {task['question']}"
    )


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

def judge_semantic(client, answer: str, gold: str, question: str) -> float:
    prompt = (
        f"Question: {question}\n\nGold hypothesis: {gold}\n\nCandidate: {answer}\n\n"
        "Does the candidate correctly capture the key relationship or finding? "
        "1=correct direction/mechanism, 0.5=partially right, 0=wrong or missing. "
        "Output only the number."
    )
    try:
        r = call_llm(client, model=STRONG, system="Strict scientific judge.",
                     user=prompt, max_tokens=5, temperature=0.0)
        return float(r.strip().split()[0])
    except Exception:
        return 0.0


def raw_answer(client, model: str, context: str) -> str:
    try:
        return call_llm(
            client, model=model,
            system="You are a precise scientific analyst.",
            user=f"{context}\n\nState your hypothesis concisely.",
            max_tokens=250, temperature=0.2,
        )
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="db_kscale_v1")
    ap.add_argument("--n_tasks", type=int, default=12)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_dir = _PROJ / "lmw" / "kscale" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"exists; --overwrite to rerun: {out_path}")

    client = make_openai_client()
    tasks = load_db_tasks(args.n_tasks)
    print(f"=== DiscoveryBench K-Scaling: {len(tasks)} tasks ===\n")

    summary: dict = {
        "run_id": args.run_id,
        "k_values": K_VALUES,
        "weak": WEAK, "strong": STRONG,
        "n_tasks": len(tasks),
        "results": {},
    }

    for task in tasks:
        name = task["name"]
        print(f"--- {name} [{task['domain']}] ---")
        ctx = build_context(task)
        gold = task["gold"]
        question = task["question"]
        res: dict = {}

        # Variation-MDL at each K
        for k in K_VALUES:
            engine = make_v2_engine(
                weak_model=WEAK, meta_model=WEAK,
                k=k, max_rounds=1, client=client,
            )
            mdl_res = engine.run(ctx)
            sem = judge_semantic(client, mdl_res.answer, gold, question)
            res[str(k)] = sem
            res[f"ratio_k{k}"] = mdl_res.compression_ratio
            print(
                f"  K={k:2d}: sem={sem} ratio={mdl_res.compression_ratio:.2f} "
                f"ans={mdl_res.answer[:60]!r}",
                flush=True,
            )

        # Strong baseline
        strong_ans = raw_answer(client, STRONG, ctx)
        res["strong_k1"] = judge_semantic(client, strong_ans, gold, question)
        print(f"  strong K=1: sem={res['strong_k1']} ans={strong_ans[:60]!r}")

        summary["results"][name] = res
        out_path.write_text(json.dumps(summary, indent=2))
        print()

    # Summary curves
    print("=== K-SCALING CURVES ===")
    for name, res in summary["results"].items():
        curve = " → ".join(
            f"K={k}:{res.get(str(k),'?'):.2f}" for k in K_VALUES
        )
        print(f"  {name}: {curve}  |  strong={res.get('strong_k1','?'):.2f}")

    # Thesis check
    print("\n=== THESIS CHECK ===")
    wins = 0
    for name, res in summary["results"].items():
        best_mdl = max((res.get(str(k), 0) for k in K_VALUES), default=0)
        strong = res.get("strong_k1", 0)
        win = best_mdl >= strong
        wins += int(win)
        print(f"  {name}: weak+MDL_best={best_mdl:.2f} vs strong={strong:.2f} → {'WIN' if win else 'loss'}")
    n = len(summary["results"])
    print(f"\n  OVERALL: {wins}/{n} weak+MDL >= strong  "
          f"({'THESIS HOLDS' if wins >= n // 2 else 'needs more work'})")

    # Calibration
    print("\n=== CALIBRATION: ratio >= 2.0 ===")
    pts = []
    for res in summary["results"].values():
        for k in K_VALUES:
            acc = res.get(str(k))
            ratio = res.get(f"ratio_k{k}")
            if acc is not None and ratio is not None:
                pts.append((float(ratio), float(acc)))
    above2 = [p for p in pts if p[0] >= 2.0]
    if above2:
        mean_acc = sum(a for _, a in above2) / len(above2)
        perfect = sum(1 for _, a in above2 if a >= 1.0)
        print(f"  ratio>=2.0: n={len(above2)} mean_acc={mean_acc:.2f} perfect={perfect}/{len(above2)}")

    summary["thesis_wins"] = f"{wins}/{n}"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nsummary → {out_path}")


if __name__ == "__main__":
    main()
