"""DiscoveryBench, fresh angle: question-TYPE-routed analysis (PAL), not forced correlation.

Diagnosis: prior attempts forced a correlation/relationship analysis on EVERY task,
but DB questions are heterogeneous (a specific period/value, which-variable,
group-comparison, relationship). The ~15 ceiling may be ANALYSIS-mismatch, not
(only) verification. Here the model SELF-ROUTES: decides what kind of finding the
question wants, writes the matching pandas code (PAL), executes it, and states the
finding in the question's framing. Isolates "is the analysis right?".

  python -m mars.runners.run_db_routed --run_id dbr_v1 --model openai/gpt-4o-mini
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import warnings
from pathlib import Path

import numpy as np
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
from mars.induction.self_portfolio import _compile      # safe pandas sandbox
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


def _json(raw):
    t = raw.strip()
    if t.startswith("```"):
        t = "\n".join(t.split("\n")[1:]).split("```")[0]
    try:
        return json.loads(t)
    except Exception:
        i, j = t.find("{"), t.rfind("}")
        try: return json.loads(t[i:j+1])
        except Exception: return {}


def route_and_analyze(client, model, q, descs, dfs):
    cols = "\n".join(f"  {c}: {d[:70]}" for c, d in list(descs.items())[:35])
    obj = _json(call_llm(client, model=model, system="Return only JSON.",
        user=(f"Discovery question: {q}\n\nColumns (name: desc):\n{cols}\n\n"
              f"FIRST decide what KIND of finding answers this question:\n"
              f"  'period'  — a specific time/period/region/value (e.g. which century X peaks)\n"
              f"  'which'   — which variable/category is responsible\n"
              f"  'compare' — a difference between groups\n"
              f"  'relation'— direction/shape between two variables\n"
              f"THEN write pandas code `def analyze(df):` that computes EXACTLY that finding and "
              f"returns a short dict of concrete results (numbers/labels). Use EXACT column names; "
              f"pd and np available; df is the table.\n"
              f'Return JSON: {{"type":"...","code":"def analyze(df):\\n    ..."}}'),
        max_tokens=700, temperature=0.3))
    code = obj.get("code", "")
    fn = _compile(code, "analyze", {})
    finding = None
    if fn is not None:
        for df in dfs:
            try:
                r = fn(df)
                if r is not None:
                    finding = r; break
            except Exception:
                continue
    return obj.get("type", "?"), code, finding


def compose(client, model, q, qtype, finding):
    hypo = call_llm(client, model=model,
        system="Write ONE precise scientific hypothesis answering the question, using ONLY the "
               "computed finding, in the question's framing. Name specific values/variables/groups.",
        user=f"Question: {q}\nFinding type: {qtype}\nComputed finding (from data):\n"
             f"{json.dumps(finding, default=str)[:500]}\n\nHypothesis (one sentence):",
        max_tokens=140, temperature=0.2)
    return hypo.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="dbr_v1")
    ap.add_argument("--model", default="openai/gpt-4o-mini")
    ap.add_argument("--judge", default="openai/gpt-4-turbo")
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out_dir = _PROJ / "lmw" / "db_routed" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit("exists; --overwrite")

    client = make_openai_client()
    tasks = {t.task_key: t for t in load_official_real_tasks(
        _PROJ / "discoverybench_repo", max_tasks=None, datasets=None, task_keys=set(args.tasks))}
    trace = io.StringIO()
    results, scores = {}, []
    print(f"=== DB question-type-routed analysis (PAL) — {args.model} ===\n")
    for key in args.tasks:
        t = tasks.get(key)
        if not t:
            continue
        descs = _col_descs(t); dfs = _all_csv(t)
        try:
            qtype, code, finding = route_and_analyze(client, args.model, t.task.query, descs, dfs)
            hypo = compose(client, args.model, t.task.query, qtype, finding) if finding is not None else ""
        except Exception as e:
            qtype, finding, hypo = "err", None, ""
        if not hypo:
            scores.append(0.0); results[key] = {"HMS": 0.0, "type": qtype, "finding": str(finding)[:120]}
            print(f"  {key:46} HMS=  0.0 [type={qtype} no finding/hypo]")
            continue
        ev = _run_official_eval(t, pred_hypo=hypo, pred_workflow="", judge_model=args.judge, trace_f=trace)
        hms = 100.0 * float(ev.get("final_score", 0.0))
        scores.append(hms)
        results[key] = {"HMS": round(hms, 1), "type": qtype, "hypo": hypo, "finding": str(finding)[:160]}
        print(f"  {key:46} HMS={hms:5.1f} [{qtype}] | {hypo[:50]}")
        out_path.write_text(json.dumps(results, indent=2, default=str))

    mean = sum(scores) / len(scores) if scores else 0.0
    print(f"\n=== RESULT (question-type-routed, {args.model}) ===")
    print(f"  mean HMS = {mean:.2f}  (N={len(scores)})")
    print(f"  prior (forced-correlation / generic, paper judge): ~14-15")
    out_path.write_text(json.dumps({"model": args.model, "mean_HMS": round(mean, 2),
                                    "results": results}, indent=2, default=str))
    (out_dir / "judge_trace.log").write_text(trace.getvalue())
    print(f"\nsummary → {out_path}")


if __name__ == "__main__":
    main()
