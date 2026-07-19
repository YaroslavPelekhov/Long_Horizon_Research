"""
Hypothesis Sketching on DiscoveryBench — the decisive test of the
division-of-labor idea against the Scaffold-Competence Law.

Idea: the small model proposes a SKETCH (target variable, predictor variable(s),
and a functional FORM from a typed menu) with numeric holes; a deterministic
solver FITS the holes against the dataframe (least-squares / curve fit) and
measures R². Refutation: if the fit is poor, the model proposes a different
sketch (different variables or form). The model never guesses numbers — it
explores STRUCTURE; the solver nails the constants.

Win condition: beat the free-form MARS-full baseline HMS = 0.680 on the same
6 DiscoveryBench tasks.

Run:
  python -m mars.srhp.sketch_db    (MARS_DB_N tasks, default 6)
"""

from __future__ import annotations

import json
import math
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


# ── typed functional forms (the sketch menu) ─────────────────────────────────
# Each maps predictor x (already selected) to a feature the solver fits linearly:
#   y ≈ a * feat(x) + b   (then R² in the fitted space)
_FORMS = {
    "linear":   lambda x: x,
    "power":    lambda x: np.sign(x) * np.log(np.abs(x) + 1e-9),   # log-log → power
    "log":      lambda x: np.log(np.abs(x) + 1e-9),
    "exp":      lambda x: x,                # y=exp: fit log(y)~x  (handled below)
    "quadratic": lambda x: x ** 2,
    "inverse":  lambda x: 1.0 / (x + 1e-9),
    "sqrt":     lambda x: np.sqrt(np.abs(x)),
}


_SKETCH_SYS = (
    "You are a scientific hypothesis SKETCHER. You do NOT compute numbers. You "
    "propose the STRUCTURE of a relationship and a numeric solver fits the "
    "constants. Given a discovery query and the dataset columns, output the "
    "single most plausible sketch as JSON:\n"
    '{"target":"<numeric outcome column>",'
    '"predictor":"<numeric predictor column>",'
    '"form":"linear|power|log|exp|quadratic|inverse|sqrt",'
    '"sign":"positive|negative"}\n'
    "Pick the columns the QUERY is about. Use EXACT column names from the list. "
    "Reply ONLY the JSON object."
)

_REVISE_SYS = (
    "Your previous sketch fit the data poorly (low R²). Propose a DIFFERENT "
    "sketch — change the predictor variable and/or the functional form — that "
    "could explain the data better. Same JSON format, exact column names. "
    "Reply ONLY the JSON object."
)


def _fit(y, x, form):
    """Fit y ≈ a*feat(x)+b for the given form; return (r2, a_sign, n)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if form in ("power",):
        m &= (x > 0) & (y > 0)
        if m.sum() < 4:
            return -1.0, 0.0, int(m.sum())
        fx, fy = np.log(x[m]), np.log(y[m])
    elif form == "exp":
        m &= (y > 0)
        if m.sum() < 4:
            return -1.0, 0.0, int(m.sum())
        fx, fy = x[m], np.log(y[m])
    else:
        if m.sum() < 4:
            return -1.0, 0.0, int(m.sum())
        fx, fy = _FORMS[form](x[m]), y[m]
    fx = np.asarray(fx, float)
    good = np.isfinite(fx) & np.isfinite(fy)
    if good.sum() < 4:
        return -1.0, 0.0, int(good.sum())
    fx, fy = fx[good], fy[good]
    A = np.vstack([fx, np.ones_like(fx)]).T
    try:
        coef, *_ = np.linalg.lstsq(A, fy, rcond=None)
    except Exception:
        return -1.0, 0.0, int(good.sum())
    pred = A @ coef
    ss_res = float(((fy - pred) ** 2).sum())
    ss_tot = float(((fy - fy.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return r2, float(coef[0]), int(good.sum())


def _numeric_cols(df):
    return list(df.select_dtypes(include="number").columns)


def sketch_episode(adapter, client, model, max_rounds=4):
    dfs = adapter.get_dataframes()
    if not dfs:
        return ""
    name = list(dfs)[0]
    df = dfs[name]
    cols = _numeric_cols(df)
    query = adapter.get_query_text()
    col_desc = adapter.get_column_descriptions()
    col_block = "\n".join(f"  {c}: {col_desc.get(c,'')[:80]}" for c in cols[:30])

    best = None  # (r2, sketch, slope, n)
    tried = []
    sysp = _SKETCH_SYS
    for rnd in range(max_rounds):
        prev = ""
        if tried:
            prev = ("\nPrevious sketches and their R² (improve on these):\n" +
                    "\n".join(f"  {json.dumps(s)} -> R2={r:.3f}" for s, r in tried))
        user = (f"DISCOVERY QUERY:\n{query}\n\nNUMERIC COLUMNS:\n{col_block}\n"
                f"{prev}\n\nOutput the sketch JSON now.")
        try:
            raw = call_llm(client, model, sysp, user, max_tokens=200, temperature=0.3)
            sk = parse_json_strict(raw) or {}
        except Exception:
            sk = {}
        tgt, prd, form = sk.get("target"), sk.get("predictor"), sk.get("form", "linear")
        if tgt not in cols or prd not in cols or tgt == prd:
            tried.append((sk, -1.0))
            sysp = _REVISE_SYS
            continue
        if form not in _FORMS:
            form = "linear"
        r2, slope, n = _fit(df[tgt].values, df[prd].values, form)
        tried.append(({"target": tgt, "predictor": prd, "form": form}, r2))
        if best is None or r2 > best[0]:
            best = (r2, {"target": tgt, "predictor": prd, "form": form,
                         "sign": ("positive" if slope >= 0 else "negative")}, slope, n)
        if r2 >= 0.9:
            break
        sysp = _REVISE_SYS

    if best is None:
        return ""
    r2, sk, slope, n = best
    direction = "positive" if slope >= 0 else "negative"
    form_phrase = {
        "linear": "a linear", "power": "a power-law", "log": "a logarithmic",
        "exp": "an exponential", "quadratic": "a quadratic",
        "inverse": "an inverse", "sqrt": "a square-root",
    }.get(sk["form"], "a")
    # narrate the fitted sketch as the hypothesis
    hypo = (
        f"The outcome '{sk['target']}' has {form_phrase} {direction} relationship "
        f"with '{sk['predictor']}' (fitted R²={r2:.2f}, n={n}). As '{sk['predictor']}' "
        f"{'increases' if direction=='positive' else 'decreases-in-effect'}, "
        f"'{sk['target']}' changes accordingly under a {sk['form']} functional form."
    )
    return hypo


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    n_tasks = int(os.environ.get("MARS_DB_N", "6"))
    model = os.environ.get("MARS_GENERATOR_MODEL", "openai/gpt-4o-mini")
    judge = os.environ.get("OLS_DB_JUDGE_MODEL", "openai/gpt-4o")
    tasks = load_db_tasks(_PROJ / "discoverybench_repo", split="train", max_tasks=n_tasks)
    client = make_openai_client()

    print(f"\n=== Hypothesis Sketching × DiscoveryBench ===")
    print(f"  model={model} judge={judge}  N={len(tasks)} tasks")
    print(f"  baseline MARS-full HMS = 0.680 (target to beat)\n")

    scores = []
    for i, t in enumerate(tasks):
        adapter = DiscoveryBenchAdapter(task=t, budget=12, judge_model=judge)
        try:
            hypo = sketch_episode(adapter, client, model)
            adapter._submitted_hypothesis = hypo
            sc = adapter.score_episode([], final_artifact=hypo)
            hms = sc.get("HMS", sc.get("primary", 0.0))
        except Exception as e:
            hms = 0.0
            hypo = f"(error: {type(e).__name__}: {e})"
        scores.append(hms)
        print(f"  [{i+1}/{len(tasks)}] {t.task_id[:50]:<50} HMS={hms:.2f}")
        print(f"        {hypo[:140]}")

    mean = sum(scores) / len(scores) if scores else 0.0
    print(f"\n  Hypothesis-Sketching HMS.mean = {mean:.3f}  (baseline 0.680)")
    verdict = "BEATS baseline" if mean > 0.680 else "below baseline"
    print(f"  → {verdict}")


if __name__ == "__main__":
    main()
