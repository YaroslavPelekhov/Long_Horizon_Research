"""
Reason-then-Verify for DiscoveryBench (open-ended competing).

The competent-baseline wall (HMS 0.68) defeated every fit-first scaffold (gate,
AutoStat, Hypothesis-Sketching) because they optimised FIT and drifted to
spurious high-R² derived columns instead of answering the QUERY.

Reason-then-Verify inverts the order:
  1. REASON  — the model reads the query + column descriptions and names the
     target + predictor the QUESTION is about (semantics first, NOT correlation),
     plus its expected direction.
  2. VERIFY  — a single targeted fit confirms the actual direction / magnitude
     of THAT pair (not a search over all pairs).
  3. COMPOSE — the hypothesis states the verified direction + coefficient.

Win condition: beat the free-form MARS-full baseline HMS = 0.680.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_PROJ = Path(__file__).resolve().parent.parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

try:
    from dotenv import load_dotenv
    _env = _PROJ / "autodiscovery" / ".env.local"
    if _env.exists():
        load_dotenv(_env, override=False)
except ImportError:
    pass

import numpy as np

from mars.adapters.discoverybench_adapter import load_db_tasks
from ols.adapters.discoverybench_adapter import DiscoveryBenchAdapter
from mars.agents.base import call_llm, make_openai_client, parse_json_strict


_REASON_SYS = (
    "You are a scientific analyst. Read the discovery QUERY and the dataset's "
    "columns. Identify the variables the QUERY is asking about — do NOT pick "
    "columns just because they might correlate; pick what the QUESTION concerns. "
    "Output JSON:\n"
    '{"target":"<outcome column the query asks to explain>",'
    '"predictor":"<the main explanatory column the query asks about>",'
    '"expected_direction":"positive|negative|unsure",'
    '"reasoning":"<one sentence tying columns to the query>"}\n'
    "Use EXACT column names. Reply ONLY the JSON."
)

_COMPOSE_SYS = (
    "Write a single-paragraph scientific hypothesis answering the discovery "
    "query. You are given the query and the VERIFIED empirical relationship "
    "(direction, coefficient, R²) between the chosen variables. State the "
    "relationship precisely with its sign and magnitude. 2-3 sentences. "
    "Reply with ONLY the hypothesis text."
)


def _verify(df, target, predictor):
    """Fit target ~ predictor (linear + power); return dict with direction etc."""
    try:
        x = np.asarray(df[predictor].values, float)
        y = np.asarray(df[target].values, float)
    except Exception:
        return None
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4:
        return None
    x, y = x[m], y[m]
    # Pearson r
    if x.std() < 1e-12 or y.std() < 1e-12:
        return None
    r = float(np.corrcoef(x, y)[0, 1])
    # linear slope
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    r2 = r * r
    direction = "positive" if r >= 0 else "negative"
    return {"direction": direction, "r": round(r, 3), "r2": round(r2, 3),
            "slope": float(slope), "n": int(m.sum())}


def reason_verify_episode(adapter, client, model):
    dfs = adapter.get_dataframes()
    if not dfs:
        return ""
    df = dfs[list(dfs)[0]]
    num_cols = list(df.select_dtypes(include="number").columns)
    query = adapter.get_query_text()
    col_desc = adapter.get_column_descriptions()
    col_block = "\n".join(f"  {c}: {col_desc.get(c, '')[:90]}" for c in num_cols[:30])

    # 1. REASON — pick query-relevant variables
    user = (f"DISCOVERY QUERY:\n{query}\n\nNUMERIC COLUMNS:\n{col_block}\n\n"
            "Identify the target and predictor the QUERY concerns. JSON now.")
    try:
        raw = call_llm(client, model, _REASON_SYS, user, max_tokens=220, temperature=0.2)
        sk = parse_json_strict(raw) or {}
    except Exception:
        sk = {}
    tgt, prd = sk.get("target"), sk.get("predictor")
    exp_dir = sk.get("expected_direction", "unsure")
    reasoning = sk.get("reasoning", "")

    # guard: must be valid distinct numeric columns
    if tgt not in num_cols or prd not in num_cols or tgt == prd:
        # fall back: let the model answer free-form (baseline behaviour)
        fb_sys = ("Answer the discovery query with a single-paragraph hypothesis "
                  "naming the key variables and the relationship. 2-3 sentences.")
        try:
            return call_llm(client, model, fb_sys,
                            f"QUERY:\n{query}\n\nCOLUMNS:\n{col_block}",
                            max_tokens=200, temperature=0.3)
        except Exception:
            return ""

    # 2. VERIFY — single targeted fit
    v = _verify(df, tgt, prd)
    if v is None:
        v = {"direction": exp_dir, "r": None, "r2": None, "slope": None, "n": 0}

    # 3. COMPOSE — grounded hypothesis
    user2 = (
        f"QUERY:\n{query}\n\n"
        f"CHOSEN VARIABLES: target='{tgt}', predictor='{prd}' "
        f"(reasoning: {reasoning})\n"
        f"VERIFIED RELATIONSHIP: direction={v['direction']}, "
        f"Pearson r={v['r']}, R²={v['r2']}, slope={v['slope']}, n={v['n']}.\n\n"
        "Write the hypothesis now."
    )
    try:
        return call_llm(client, model, _COMPOSE_SYS, user2, max_tokens=220, temperature=0.3)
    except Exception:
        return (f"'{tgt}' has a {v['direction']} relationship with '{prd}' "
                f"(r={v['r']}, n={v['n']}).")


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    n = int(os.environ.get("MARS_DB_N", "6"))
    reps = int(os.environ.get("MARS_DB_REPS", "1"))
    model = os.environ.get("MARS_GENERATOR_MODEL", "openai/gpt-4o-mini")
    judge = os.environ.get("OLS_DB_JUDGE_MODEL", "openai/gpt-4o")
    tasks = load_db_tasks(_PROJ / "discoverybench_repo", split="train", max_tasks=n)
    client = make_openai_client()

    print(f"\n=== Reason-then-Verify × DiscoveryBench ===")
    print(f"  model={model} judge={judge}  N={len(tasks)} x {reps} reps")
    print(f"  baseline MARS-full HMS = 0.680 (target to beat)\n")

    all_scores = []
    for rep in range(reps):
        scores = []
        for i, t in enumerate(tasks):
            adapter = DiscoveryBenchAdapter(task=t, budget=12, judge_model=judge)
            try:
                hypo = reason_verify_episode(adapter, client, model)
                adapter._submitted_hypothesis = hypo
                sc = adapter.score_episode([], final_artifact=hypo)
                hms = sc.get("HMS", 0.0)
            except Exception as e:
                hms, hypo = 0.0, f"(err {e})"
            scores.append(hms)
            if rep == 0:
                print(f"  [{i+1}/{len(tasks)}] {t.task_id[-28:]:<28} HMS={hms:.2f}  {str(hypo)[:90]}")
        all_scores.extend(scores)
        m = sum(scores) / len(scores) if scores else 0.0
        print(f"  rep{rep} mean={m:.3f}")

    mean = sum(all_scores) / len(all_scores) if all_scores else 0.0
    print(f"\n  Reason-then-Verify HMS.mean = {mean:.3f}  (baseline 0.680)")
    print(f"  → {'BEATS baseline' if mean > 0.680 else 'below/at baseline'}")


if __name__ == "__main__":
    main()
