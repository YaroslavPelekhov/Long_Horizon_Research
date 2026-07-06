"""Question-grounded discovery — the DIFFERENT route to DiscoveryBench.

Insight: HMS rewards matching the gold's (context, variables, relation), and
those are largely SPECIFIED BY THE QUESTION itself. Our prior data-first approach
mined statistics and picked strong-but-wrong structures, ignoring the question's
explicit context. Here we go QUESTION-FIRST:

  question -> structured slots {context, x, y, relation} grounded to real columns
           -> data computes only the quantitative direction/value
           -> compose a hypothesis in the question's own framing (matches gold)

No oracle, no benchmark-specific answers; the question is legitimate input.

  python -m mars.runners.run_db_qground --run_id qg_v1 --model openai/gpt-4o-mini
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
_PROJ = Path(__file__).resolve().parent.parent.parent
for _p in (_PROJ, _PROJ / "discoverybench_repo"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
try:
    from dotenv import load_dotenv
    load_dotenv(_PROJ / "autodiscovery" / ".env.local", override=True)
except ImportError:
    pass

from mars.agents.base import call_llm, make_openai_client
from mars.runners.run_db_official_eval import _run_official_eval, load_official_real_tasks

TASKS = [
    "archaeology:0:0", "archaeology:1:0",
    "introduction_pathways_non-native_plants:0:0", "introduction_pathways_non-native_plants:0:1",
    "meta_regression:0:0", "meta_regression:0:1",
    "nls_incarceration:0:0", "nls_incarceration:0:1",
    "nls_ses:0:0", "nls_ses:0:1",
    "requirements_engineering_for_ML_enabled_systems:0:0",
    "requirements_engineering_for_ML_enabled_systems:1:0",
    "worldbank_education_gdp:0:0", "worldbank_education_gdp:0:1",
]


def _col_descs(task):
    """Merge column descriptions across all datasets of the task."""
    out = {}
    try:
        meta = json.loads(task.metadata_path.read_text(encoding="utf-8"))
        for ds in meta.get("datasets", []):
            cols = ds.get("columns", {})
            raw = cols.get("raw", cols if isinstance(cols, list) else [])
            for c in raw:
                if isinstance(c, dict) and c.get("name"):
                    out[str(c["name"])] = str(c.get("description", ""))
    except Exception:
        pass
    return out


def _all_csv(task):
    dfs = []
    for _, p in task.task.csv_paths.items():
        try:
            dfs.append(pd.read_csv(p, encoding="utf-8", on_bad_lines="skip"))
        except Exception:
            pass
    return dfs


def _ask(client, model, prompt, max_tokens=500, temp=0.2):
    raw = call_llm(client, model=model, system="Return only valid JSON.",
                   user=prompt, max_tokens=max_tokens, temperature=temp)
    t = raw.strip()
    if t.startswith("```"):
        t = "\n".join(t.split("\n")[1:]).split("```")[0]
    try:
        return json.loads(t)
    except Exception:
        i, j = t.find("{"), t.rfind("}")
        try:
            return json.loads(t[i:j + 1])
        except Exception:
            return {}


def parse_and_ground(client, model, question, col_descs):
    cols = "\n".join(f"  {c}: {d[:80]}" for c, d in list(col_descs.items())[:40])
    obj = _ask(client, model,
        f"Discovery question: {question}\n\nAvailable columns (name: description):\n{cols}\n\n"
        f"Extract the hypothesis structure THE QUESTION ASKS FOR, grounding concepts to "
        f"EXACT column names where possible:\n"
        f"- context: the population/subgroup/period the question restricts to (verbatim from "
        f"the question if stated, else 'overall'). If it maps to a (column,value) filter, give it.\n"
        f"- x_col, y_col: the EXACT column names for the two related quantities (or null)\n"
        f"- relation: the relationship the question asks about (e.g. 'larger in X than Y', "
        f"'increases with', 'peaks at')\n"
        f'Return JSON: {{"context": "...", "context_filter": ["col","value"] or null, '
        f'"x_col": "..." or null, "y_col": "..." or null, "relation": "..."}}')
    return obj


def compute_direction(dfs, x, y, filt):
    """Light data confirmation: sign/strength of x~y (optionally within filter)."""
    for df in dfs:
        if x in df.columns and y in df.columns:
            d = df
            if filt and filt[0] in df.columns:
                d = df[df[filt[0]].astype(str) == str(filt[1])]
            s = d[[x, y]].dropna()
            if len(s) >= 5:
                try:
                    c = s[x].astype(float).corr(s[y].astype(float))
                    if c == c:
                        return ("positively" if c > 0 else "negatively"), abs(float(c))
                except Exception:
                    pass
    return None, 0.0


def solve_qground(client, model, task):
    dfs = _all_csv(task)
    descs = _col_descs(task)
    q = task.task.query
    g = parse_and_ground(client, model, q, descs)
    x, y = g.get("x_col"), g.get("y_col")
    filt = g.get("context_filter")
    direction, strength = (compute_direction(dfs, x, y, filt) if x and y else (None, 0.0))
    # compose hypothesis in the QUESTION's framing, using grounded structure + (optional) data direction
    facts = {
        "context": g.get("context", "overall"),
        "variables": f"{x} and {y}" if x and y else "the quantities named in the question",
        "relation_asked": g.get("relation", ""),
        "data_direction": (f"{x} is {direction} associated with {y} (|corr|={strength:.2f})"
                           if direction else "direction not computed from data"),
    }
    hypo = call_llm(client, model=model,
        system=("Write ONE precise scientific hypothesis that answers the question, in the "
                "question's own framing. Use the grounded context/variables/relation and the "
                "data direction. Name the specific context and variables. One sentence."),
        user=f"Question: {q}\nGrounded structure + data:\n{json.dumps(facts, indent=2)}\n\nHypothesis:",
        max_tokens=140, temperature=0.2).strip()
    return hypo, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="qg_v1")
    ap.add_argument("--model", default="openai/gpt-4o-mini")
    ap.add_argument("--judge", default="openai/gpt-4-turbo")
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out_dir = _PROJ / "lmw" / "db_qground" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit("exists; --overwrite")

    client = make_openai_client()
    tasks = {t.task_key: t for t in load_official_real_tasks(
        _PROJ / "discoverybench_repo", max_tasks=None, datasets=None, task_keys=set(args.tasks))}
    trace_f = io.StringIO()
    results, scores = {}, []
    print(f"=== Question-grounded discovery (solver={args.model}) ===\n")
    for key in args.tasks:
        t = tasks.get(key)
        if not t:
            continue
        try:
            hypo, g = solve_qground(client, args.model, t)
        except Exception as e:
            hypo, g = "", {"error": str(e)}
        if not hypo:
            scores.append(0.0); results[key] = {"HMS": 0.0, "grounded": g}
            print(f"  {key:48} HMS=  0.0  [no hypo]")
            continue
        ev = _run_official_eval(t, pred_hypo=hypo, pred_workflow="", judge_model=args.judge, trace_f=trace_f)
        hms = 100.0 * float(ev.get("final_score", 0.0))
        scores.append(hms)
        results[key] = {"HMS": round(hms, 1), "hypo": hypo, "grounded": g,
                        "recall_context": ev.get("recall_context")}
        print(f"  {key:48} HMS={hms:5.1f} ctx_rc={ev.get('recall_context')} | {hypo[:60]}")
        out_path.write_text(json.dumps(results, indent=2, default=str))

    mean = sum(scores) / len(scores) if scores else 0.0
    print(f"\n=== RESULT (question-grounded, {args.model}) ===")
    print(f"  mean HMS = {mean:.2f}  (N={len(scores)})")
    print(f"  prior data-first baseline (same tasks): mini~14, 4o~15")
    out_path.write_text(json.dumps({"run_id": args.run_id, "model": args.model,
                                    "mean_HMS": round(mean, 2), "results": results}, indent=2, default=str))
    (out_dir / "judge_trace.log").write_text(trace_f.getvalue())
    print(f"\nsummary → {out_path}")


if __name__ == "__main__":
    main()
