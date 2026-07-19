"""Measure the amplification thesis on DiscoveryBench:

    weak (gpt-4o-mini) + EVA   >=?   strong (gpt-4o) raw

Three conditions, same tasks, same official HMS judge:
  A. weak_raw   : gpt-4o-mini, single shot hypothesis
  B. strong_raw : gpt-4o, single shot hypothesis
  C. weak_eva   : gpt-4o-mini wrapped in EVA (task-agnostic amplifier)

EVA gets ONE universal tool: execute(python) with the task's DataFrame bound in.
EVA contains no DiscoveryBench knowledge — the same loop would run on any task.

Run:
  python -m mars.runners.run_eva_amplification_db \\
    --run_id eva_db_v1 --tasks worldbank_education_gdp:3:0,nls_ses:0:0 \\
    --weak openai/gpt-4o-mini --strong openai/gpt-4o --judge openai/gpt-4o --overwrite
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

_PROJ = Path(__file__).resolve().parent.parent.parent
for _p in (_PROJ, _PROJ / "discoverybench_repo"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    from dotenv import load_dotenv
    _env = _PROJ / "autodiscovery" / ".env.local"
    if _env.exists():
        load_dotenv(_env, override=False)
except ImportError:
    pass

import pandas as pd  # noqa: E402

from mars.agents.base import call_llm, make_openai_client  # noqa: E402
from mars.amplify.eva import EVA, ExecResult  # noqa: E402
from mars.runners.run_db_official_eval import load_official_real_tasks, _run_official_eval  # noqa: E402

_DB_TEST = _PROJ / "discoverybench_repo" / "discoverybench" / "real" / "test"


# ---------------------------------------------------------------------------
# Task loading
# ---------------------------------------------------------------------------

def load_task(task_key: str) -> dict[str, Any]:
    dataset, mid, qid = task_key.split(":")
    meta = json.load(open(_DB_TEST / dataset / f"metadata_{mid}.json"))
    q = meta["queries"][int(qid)][0]
    ds = meta["datasets"][0]
    # choose richest dataset
    best = None
    for d in meta["datasets"]:
        p = _DB_TEST / dataset / d["name"]
        if p.exists():
            try:
                df = pd.read_csv(p)
            except Exception:
                continue
            for c in df.columns:
                if df[c].dtype == object:
                    try:
                        df[c] = df[c].str.replace(",", ".", regex=False).astype(float)
                    except Exception:
                        pass
            cd = {col["name"]: col["description"]
                  for col in d.get("columns", {}).get("raw", []) if col.get("name")}
            score = len(cd) * 2 + min(len(df), 1000)
            if best is None or score > best[3]:
                best = (df, d["name"], cd, score)
    df, dname, col_descs = best[0], best[1], best[2]
    return {
        "task_key": task_key, "dataset": dataset,
        "question": q["question"], "domain": meta.get("domain_knowledge", "")[:600],
        "df": df, "df_name": dname, "col_descs": col_descs,
    }


def task_prompt(t: dict) -> str:
    cols = "\n".join(f"  '{c}'{' — ' + t['col_descs'][c] if c in t['col_descs'] else ''}"
                     for c in t["df"].columns)
    return (
        f"Answer this scientific research question with a precise 1-2 sentence hypothesis.\n\n"
        f"QUESTION: {t['question']}\n\n"
        f"DOMAIN CONTEXT: {t['domain'][:400]}\n\n"
        f"A pandas DataFrame `df` (shape {t['df'].shape}) is available with columns:\n{cols}\n\n"
        f"Your final answer must be a specific hypothesis naming the relevant variables, "
        f"direction of effect, and any subgroup/period — grounded in the data."
    )


# ---------------------------------------------------------------------------
# Universal sandbox (the ONE tool EVA uses)
# ---------------------------------------------------------------------------

def make_executor(df: pd.DataFrame):
    import numpy as np

    def execute(code: str) -> ExecResult:
        ns: dict[str, Any] = {
            "pd": pd, "np": np, "df": df.copy(),
            "__builtins__": __builtins__,
        }
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                exec(code, ns)
            return ExecResult(ok=True, value=ns.get("result"), stdout=buf.getvalue())
        except Exception as e:
            return ExecResult(ok=False, error=f"{type(e).__name__}: {e}", stdout=buf.getvalue())

    return execute


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

def raw_hypothesis(client, model: str, t: dict) -> str:
    return call_llm(
        client, model=model,
        system="You are a scientist. Answer with a precise, data-grounded hypothesis.",
        user=task_prompt(t), max_tokens=300, temperature=0.2,
    )


def score_hms(task_key: str, dataset: str, hypothesis: str, judge: str) -> float:
    tasks = load_official_real_tasks(_PROJ / "discoverybench_repo", max_tasks=None,
                                     datasets={dataset}, task_keys={task_key})
    if not tasks:
        return 0.0
    ev = _run_official_eval(tasks[0], pred_hypo=hypothesis, pred_workflow="",
                            judge_model=judge, trace_f=io.StringIO())
    return 100.0 * float(ev.get("final_score", 0.0))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="eva_db_smoke")
    ap.add_argument("--tasks", default="worldbank_education_gdp:3:0")
    ap.add_argument("--weak", default="openai/gpt-4o-mini")
    ap.add_argument("--strong", default="openai/gpt-4o")
    ap.add_argument("--judge", default="openai/gpt-4o")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out_dir = _PROJ / "lmw" / "eva" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"exists; pass --overwrite: {out_path}")

    client = make_openai_client()
    task_keys = [k.strip() for k in args.tasks.split(",") if k.strip()]
    rows = []

    print(f"=== Amplification test: weak+EVA vs strong on {len(task_keys)} DB tasks ===")
    print(f"weak={args.weak}  strong={args.strong}  judge={args.judge}\n")

    for tk in task_keys:
        t = load_task(tk)
        print(f"[{tk}] {t['question'][:70]}")

        # A. weak raw
        h_weak = raw_hypothesis(client, args.weak, t)
        s_weak = score_hms(tk, t["dataset"], h_weak, args.judge)

        # B. strong raw
        h_strong = raw_hypothesis(client, args.strong, t)
        s_strong = score_hms(tk, t["dataset"], h_strong, args.judge)

        # C. weak + EVA (task-agnostic)
        eva = EVA(model=args.weak, k_candidates=args.k, max_rounds=args.rounds)
        execute = make_executor(t["df"])
        exec_contract = ("runs in a Python sandbox where `df` (the dataset), pd, np, "
                         "and CANDIDATE_ANSWER (str hypothesis) are available; "
                         "recompute statistics from df to verify/refute the hypothesis; "
                         "print PASS or FAIL: <reason>")
        eva_res = eva.solve(task_prompt(t), execute, exec_contract=exec_contract)
        s_eva = score_hms(tk, t["dataset"], eva_res.answer, args.judge)

        amplified = s_eva >= s_strong
        print(f"   weak_raw={s_weak:5.1f}   strong_raw={s_strong:5.1f}   "
              f"weak+EVA={s_eva:5.1f}   {'✓ amplified' if amplified else '✗'}")

        rows.append({
            "task_key": tk, "question": t["question"][:100],
            "weak_raw": s_weak, "strong_raw": s_strong, "weak_eva": s_eva,
            "amplified": amplified,
            "eva": eva_res.to_dict(),
            "hyp_weak": h_weak[:200], "hyp_strong": h_strong[:200], "hyp_eva": eva_res.answer[:200],
        })

    n = len(rows)
    mw = sum(r["weak_raw"] for r in rows) / n
    ms = sum(r["strong_raw"] for r in rows) / n
    me = sum(r["weak_eva"] for r in rows) / n
    summary = {
        "run_id": args.run_id, "score_type": "amplification-weak+eva-vs-strong",
        "weak": args.weak, "strong": args.strong, "judge": args.judge,
        "n_tasks": n,
        "mean_weak_raw": round(mw, 1), "mean_strong_raw": round(ms, 1),
        "mean_weak_eva": round(me, 1),
        "amplified_vs_strong": me >= ms,
        "n_tasks_eva_beats_strong": sum(1 for r in rows if r["amplified"]),
        "results": rows,
    }
    out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print(f"\n=== MEAN ===")
    print(f"  weak_raw   : {mw:.1f}")
    print(f"  strong_raw : {ms:.1f}")
    print(f"  weak+EVA   : {me:.1f}   {'>= strong ✓ THESIS HOLDS' if me >= ms else '< strong'}")
    print(f"  EVA beats strong on {summary['n_tasks_eva_beats_strong']}/{n} tasks")
    print(f"summary → {out_path}")


if __name__ == "__main__":
    main()
