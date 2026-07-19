"""DiscoveryBench via CONSERVATION-LAW selection (inference invariants).

Principle (Noether for inference): the truth of a quantitative claim is INVARIANT
under transformations that preserve the data's meaning — bootstrap resampling, row
permutation, disjoint subsampling. A CORRECT answer's central quantity is CONSERVED
across these; a SPURIOUS one is not (it evaporates / flips sign). We select the
candidate whose claimed quantity is most conserved AND that discriminates against
the model's own negatives; else we abstain.

The model authors everything: K candidate (answer, measure) pairs, where measure(ctx)
returns the SIGNED scalar the answer asserts. The shell only applies truth-preserving
transformations and checks conservation. No DB-specific code; judge is the only
benchmark-touching part (official HMS with the paper judge gpt-4-turbo).

  python -m mars.runners.run_db_invariants --run_id inv_v1 --model openai/gpt-4o-mini
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
    load_dotenv(_PROJ / "autodiscovery" / ".env.local", override=False)
except ImportError:
    pass

from mars.agents.base import call_llm, make_openai_client
from mars.induction.self_portfolio import _compile
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


def _largest(task):
    best, n = None, -1
    for _, p in task.task.csv_paths.items():
        try:
            df = pd.read_csv(p, encoding="utf-8", on_bad_lines="skip")
            if len(df) > n:
                best, n = df, len(df)
        except Exception:
            pass
    return best


def _llm_json(client, model, prompt, max_tokens=2000, temp=0.5):
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


def author_candidates(client, model, question, ctx_desc, k=5):
    """Model authors K (answer, measure) pairs. measure(ctx) returns the SIGNED
    scalar quantity the answer's central claim asserts (e.g. correlation, slope,
    group difference). Conservation of this quantity across resampling tests truth."""
    obj = _llm_json(client, model,
        f"QUESTION:\n{question}\n\nDATA INTERFACE (ctx given to your function):\n{ctx_desc}\n\n"
        f"Propose {k} DIVERSE candidate answers to the question. For EACH, also write a "
        f"`def measure(ctx):` that returns the SINGLE SIGNED scalar that this answer's central "
        f"claim asserts — the quantity that, if the claim is TRUE, will stay stable in sign and "
        f"magnitude when the data is resampled. Examples: a claim of positive association -> return "
        f"the correlation; a claim that group A > group B -> return mean(A)-mean(B); a claim of a "
        f"trend slope -> return the regression slope. Use ctx['df'] (pandas), ctx['columns']. "
        f"The SIGN of measure must match the answer's claimed direction.\n"
        f'Return JSON: {{"candidates":[{{"answer":"...","measure":"def measure(ctx):\\n    ..."}}]}}',
        max_tokens=2400, temp=0.6)
    out = []
    for c in obj.get("candidates", []):
        if isinstance(c, dict) and c.get("answer") and c.get("measure"):
            out.append((str(c["answer"]), str(c["measure"])))
    return out[:k]


def author_negatives_measure(client, model, question, answer):
    """Negatives = plausible-but-wrong variants of the SAME claim (corrupt direction/
    variable/group). A faithful conserved quantity must SEPARATE answer from these."""
    obj = _llm_json(client, model,
        f"QUESTION:\n{question}\n\nCANDIDATE ANSWER:\n{answer}\n\n"
        f"Give 3 PLAUSIBLE-BUT-WRONG variants (flip the direction, swap the variable/group). "
        f'Return JSON: {{"negatives":["...","..."]}}', max_tokens=400, temp=0.4)
    return [str(n) for n in obj.get("negatives", []) if str(n).strip()][:3]


def _safe_measure(fn, ctx):
    try:
        v = float(fn(dict(ctx)))
        return v if np.isfinite(v) else None
    except Exception:
        return None


def conservation_score(fn, df, n_resamples=20, seed=0):
    """Apply truth-preserving transformations; return (sign_consistency, stability, full_val).
    sign_consistency = fraction of resamples whose measure has the SAME sign as the full-data
    measure. stability = 1/(1+CV) where CV is coefficient of variation across resamples.
    A conserved quantity has high sign_consistency AND high stability."""
    rng = np.random.RandomState(seed)
    full = _safe_measure(fn, {"df": df, "columns": list(df.columns)})
    if full is None or full == 0:
        return 0.0, 0.0, full
    vals = []
    n = len(df)
    for i in range(n_resamples):
        if i % 2 == 0:                                   # bootstrap resample
            idx = rng.randint(0, n, size=n)
            sub = df.iloc[idx].reset_index(drop=True)
        else:                                            # disjoint ~60% subsample + row shuffle
            idx = rng.permutation(n)[: max(3, int(n * 0.6))]
            sub = df.iloc[idx].reset_index(drop=True)
        v = _safe_measure(fn, {"df": sub, "columns": list(sub.columns)})
        if v is not None:
            vals.append(v)
    if len(vals) < n_resamples * 0.5:
        return 0.0, 0.0, full
    vals = np.array(vals)
    sign_consistency = float(np.mean(np.sign(vals) == np.sign(full)))
    cv = float(np.std(vals) / (abs(np.mean(vals)) + 1e-12))
    stability = 1.0 / (1.0 + cv)
    return sign_consistency, stability, full


def discrimination_z(fn, df, n_null=30, seed=1):
    """Permutation null: shuffle each column INDEPENDENTLY (destroys all cross-column
    structure). A REAL relationship collapses toward the null; an artifact/marginal does
    not move. Return z = (|full| - mean|null|) / std|null|. High z => relationship is real,
    not a spurious-but-stable quantity. This is the discrimination the pure-conservation
    test lacks (stability != relevance)."""
    rng = np.random.RandomState(seed)
    full = _safe_measure(fn, {"df": df, "columns": list(df.columns)})
    if full is None:
        return 0.0
    nulls = []
    n = len(df)
    for _ in range(n_null):
        shuffled = df.copy()
        for col in shuffled.columns:
            shuffled[col] = shuffled[col].values[rng.permutation(n)]
        v = _safe_measure(fn, {"df": shuffled, "columns": list(shuffled.columns)})
        if v is not None:
            nulls.append(abs(v))
    if len(nulls) < n_null * 0.5:
        return 0.0
    nulls = np.array(nulls)
    return float((abs(full) - np.mean(nulls)) / (np.std(nulls) + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="inv_v1")
    ap.add_argument("--model", default="openai/gpt-4o-mini")
    ap.add_argument("--judge", default="openai/gpt-4-turbo")
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--sign_thresh", type=float, default=0.9)
    ap.add_argument("--stab_thresh", type=float, default=0.5)
    ap.add_argument("--z_thresh", type=float, default=2.0,
                    help="permutation-null z: keep only candidates whose quantity is real")
    ap.add_argument("--discriminate", action="store_true",
                    help="add permutation-null discrimination on top of conservation")
    ap.add_argument("--baseline", action="store_true",
                    help="control: always answer with first candidate, no conservation gate")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out_dir = _PROJ / "lmw" / "db_invariants" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit("exists; --overwrite")

    client = make_openai_client()
    tasks = {t.task_key: t for t in load_official_real_tasks(
        _PROJ / "discoverybench_repo", max_tasks=None, datasets=None, task_keys=set(args.tasks))}
    trace = io.StringIO()
    results, answered = {}, []
    n_ans = n_abs = 0
    print(f"=== DiscoveryBench — Conservation-law selection — {args.model} ===\n")
    for key in args.tasks:
        t = tasks.get(key)
        if not t:
            continue
        df = _largest(t)
        if df is None:
            n_abs += 1; results[key] = {"status": "ABSTAIN", "note": "no csv"}
            print(f"  {key:46} ABSTAIN (no csv)"); continue
        ctx_desc = ("ctx['df']: pandas DataFrame; ctx['columns']: column names. "
                    "Columns: " + ", ".join(map(str, df.columns[:25])))
        try:
            cands = author_candidates(client, args.model, t.task.query, ctx_desc, args.k)
        except Exception as e:
            n_abs += 1; results[key] = {"status": "ERROR", "note": str(e)[:90]}
            print(f"  {key:46} ERROR {str(e)[:40]}"); continue

        if args.baseline:
            # control: take the model's FIRST candidate, always answer (no selection)
            if not cands:
                n_abs += 1; results[key] = {"status": "ABSTAIN", "note": "no candidate"}
                print(f"  {key:46} ABSTAIN (no candidate)"); continue
            ev = _run_official_eval(t, pred_hypo=cands[0][0], pred_workflow="",
                                    judge_model=args.judge, trace_f=trace)
            hms = 100.0 * float(ev.get("final_score", 0.0))
            answered.append(hms); n_ans += 1
            results[key] = {"status": "ANSWER", "HMS": round(hms, 1), "answer": cands[0][0][:120]}
            print(f"  {key:46} HMS={hms:5.1f} [baseline-first] | {cands[0][0][:42]}")
            out_path.write_text(json.dumps(results, indent=2, default=str)); continue

        scored = []
        for ans, mcode in cands:
            fn = _compile(mcode, "measure", {})
            if fn is None:
                continue
            sign_c, stab, full = conservation_score(fn, df)
            z = discrimination_z(fn, df) if args.discriminate else None
            scored.append({"answer": ans, "sign": round(sign_c, 3), "stab": round(stab, 3),
                           "full": None if full is None else round(full, 4),
                           "z": None if z is None else round(z, 2),
                           "score": round(sign_c * stab, 4)})

        # keep only conserved candidates (truth-preserving stability)
        conserved = [c for c in scored if c["sign"] >= args.sign_thresh and c["stab"] >= args.stab_thresh]
        # + discrimination: the conserved quantity must be a REAL relationship (outside null)
        if args.discriminate:
            conserved = [c for c in conserved if c["z"] is not None and c["z"] >= args.z_thresh]
        if not conserved:
            n_abs += 1
            results[key] = {"status": "ABSTAIN", "note": "no conserved quantity",
                            "candidates": scored}
            print(f"  {key:46} ABSTAIN (nothing conserved; best={max([c['score'] for c in scored], default=0):.2f})")
            out_path.write_text(json.dumps(results, indent=2, default=str)); continue

        best = max(conserved, key=lambda c: c["score"])
        ev = _run_official_eval(t, pred_hypo=best["answer"], pred_workflow="",
                                judge_model=args.judge, trace_f=trace)
        hms = 100.0 * float(ev.get("final_score", 0.0))
        answered.append(hms); n_ans += 1
        results[key] = {"status": "ANSWER", "HMS": round(hms, 1), "sign": best["sign"],
                        "stab": best["stab"], "score": best["score"], "answer": best["answer"][:120],
                        "n_conserved": len(conserved), "n_cands": len(scored)}
        print(f"  {key:46} HMS={hms:5.1f} cons={best['score']:.2f} ({len(conserved)}/{len(scored)}) | {best['answer'][:42]}")
        out_path.write_text(json.dumps(results, indent=2, default=str))

    n = n_ans + n_abs
    prec = sum(answered) / len(answered) if answered else 0.0
    forced = sum(answered) / n if n else 0.0
    print(f"\n=== RESULT (N={n}) ===")
    print(f"  answered {n_ans}/{n}  abstained {n_abs}/{n}")
    print(f"  PRECISION on answered = {prec:.2f} HMS")
    print(f"  forced (abstain=0)    = {forced:.2f} HMS")
    print(f"  baselines: MARS-full=33.4 | MARS-all-off=40.8 | selfverify-precision ~ ? | SOTA 24.5")
    out_path.write_text(json.dumps({"model": args.model, "answered": n_ans, "abstained": n_abs,
                                    "precision_answered": round(prec, 2), "forced_mean": round(forced, 2),
                                    "results": results}, indent=2, default=str))
    (out_dir / "judge_trace.log").write_text(trace.getvalue())
    print(f"\nsummary → {out_path}")


if __name__ == "__main__":
    main()
