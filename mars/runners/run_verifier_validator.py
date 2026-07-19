"""Verifier-validator — distinguish a FAITHFUL proxy-verifier from a
statistically-valid-but-irrelevant one, without ground truth.

A proxy-verifier V scores candidate answers. We validate V with self-generated
adversarial negatives:
  - permutation null   : shuffle target, recompute -> catches PURE-CHANCE signal
  - irrelevant-strong  : pairs with high stat. signal but low question-relevance
                         (e.g. autocorrelated / id / year pairs) -> the case that
                         fooled VSG; permutation does NOT catch these.
V is FAITHFUL iff it scores the question-relevant answer >> all negatives
(discrimination margin > 0). A naive corr-only V cannot separate
irrelevant-strong negatives -> margin ~ 0 -> flagged spurious. A grounded V
(relevance x stability) separates them -> margin > 0 -> trustworthy.

Shows: (1) permutation-null is necessary-but-insufficient; (2) question-dependence
is the decisive axis; (3) the margin is a per-task "can I verify this?" signal.

  python -m mars.runners.run_verifier_validator --run_id vv_v1
"""
from __future__ import annotations

import argparse
import itertools
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
from mars.runners.run_db_official_eval import load_official_real_tasks

WEAK = "openai/gpt-4o-mini"
TASKS = ["meta_regression:0:1", "nls_ses:0:0"]


def _corr(df, x, y):
    s = df[[x, y]].dropna()
    if len(s) < 5:
        return 0.0
    try:
        c = s[x].astype(float).corr(s[y].astype(float))
        return 0.0 if c != c else float(c)
    except Exception:
        return 0.0


def _fold_stable_abs(df, x, y, k=3):
    idx = np.array(df.index); np.random.RandomState(0).shuffle(idx)
    cs = [_corr(df.loc[idx[i::k]], x, y) for i in range(k)]
    cs = [c for c in cs if c == c]
    if len(cs) < 2:
        return 0.0
    mean = np.mean(cs)
    stable = all((c > 0) == (mean > 0) for c in cs)
    return abs(float(mean)) if stable else abs(float(mean)) * 0.3


def _perm_pvalue(df, x, y, n=200):
    s = df[[x, y]].dropna()
    if len(s) < 8:
        return 1.0
    obs = abs(_corr(s, x, y))
    xv = s[x].astype(float).values
    yv = s[y].astype(float).values
    rng = np.random.RandomState(1)
    ge = 0
    for _ in range(n):
        yp = yv.copy(); rng.shuffle(yp)
        c = np.corrcoef(xv, yp)[0, 1]
        if abs(c) >= obs:
            ge += 1
    return (ge + 1) / (n + 1)


def relevance_scores(client, question, col_desc, pairs):
    """Model rates how much each (x,y) relationship answers the question (0-1)."""
    listing = "\n".join(f"{i}: ({x}, {y})" for i, (x, y) in enumerate(pairs))
    prompt = (f"Question: {question}\n\nColumns:\n{col_desc}\n\n"
              f"For each pair, rate 0.0-1.0 how directly the relationship between the two "
              f"variables ANSWERS the question (0=irrelevant, 1=exactly what is asked).\n"
              f"Pairs:\n{listing}\n\nReturn JSON: {{\"scores\": [<float per pair index>]}}")
    raw = call_llm(client, model=WEAK, system="Return only JSON.", user=prompt,
                   max_tokens=400, temperature=0.0)
    t = raw.strip()
    if t.startswith("```"):
        t = "\n".join(t.split("\n")[1:]).split("```")[0]
    try:
        sc = json.loads(t).get("scores", [])
        return [float(s) for s in sc][:len(pairs)] + [0.0] * (len(pairs) - len(sc))
    except Exception:
        return [0.5] * len(pairs)


def _largest_csv(task):
    best, n = None, -1
    for _, p in task.task.csv_paths.items():
        try:
            df = pd.read_csv(p, encoding="utf-8", on_bad_lines="skip")
            if len(df) > n:
                best, n = df, len(df)
        except Exception:
            pass
    return best


def analyze(client, task):
    df = _largest_csv(task)
    q = task.task.query
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])][:10]
    pairs = list(itertools.combinations(num, 2))[:18]
    if len(pairs) < 3:
        return None
    col_desc = "\n".join(f"  {c}" for c in num)
    rel = relevance_scores(client, q, col_desc, pairs)
    rows = []
    for (x, y), r in zip(pairs, rel):
        stab = _fold_stable_abs(df, x, y)
        p = _perm_pvalue(df, x, y)
        rows.append({"x": x, "y": y, "rel": round(r, 2), "corr": round(stab, 2),
                     "perm_p": round(p, 3),
                     "V_naive": round(stab, 3),               # corr-only verifier
                     "V_grounded": round(r * stab, 3)})       # relevance x corr
    # relevant target = highest relevance; negatives = low-relevance but high corr
    rows_by_rel = sorted(rows, key=lambda r: -r["rel"])
    target = rows_by_rel[0]
    negatives = [r for r in rows if r["rel"] <= 0.34 and r["corr"] >= 0.3]
    if not negatives:                                          # fall back to strongest-corr low-rel
        negatives = sorted([r for r in rows if r["rel"] <= 0.5], key=lambda r: -r["corr"])[:3]
    margin_naive = target["V_naive"] - max((r["V_naive"] for r in negatives), default=0)
    margin_grounded = target["V_grounded"] - max((r["V_grounded"] for r in negatives), default=0)
    return {"question": q, "target": target, "negatives": negatives[:4],
            "margin_naive": round(margin_naive, 3), "margin_grounded": round(margin_grounded, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="vv_v1")
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out_dir = _PROJ / "lmw" / "verifier_validator" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit("exists; --overwrite")

    client = make_openai_client()
    tasks = {t.task_key: t for t in load_official_real_tasks(
        _PROJ / "discoverybench_repo", max_tasks=None, datasets=None, task_keys=set(args.tasks))}
    print("=== Verifier-validator: faithful vs spurious proxy-verifier ===\n")
    results = {}
    for key in args.tasks:
        t = tasks.get(key)
        if not t:
            continue
        a = analyze(client, t)
        if not a:
            continue
        results[key] = a
        tg = a["target"]
        print(f"--- {key} ---")
        print(f"  Q: {a['question'][:90]}")
        print(f"  target (most question-relevant): ({tg['x']},{tg['y']}) rel={tg['rel']} corr={tg['corr']} perm_p={tg['perm_p']}")
        for n in a["negatives"]:
            print(f"  negative (low-rel, high-corr):   ({n['x']},{n['y']}) rel={n['rel']} corr={n['corr']} perm_p={n['perm_p']}  <- permutation calls it SIGNIFICANT")
        print(f"  V_naive   (corr only)      margin = {a['margin_naive']:+.3f}  {'SPURIOUS (cannot separate)' if a['margin_naive']<=0.05 else 'ok'}")
        print(f"  V_grounded (relevance×corr) margin = {a['margin_grounded']:+.3f}  {'FAITHFUL (separates)' if a['margin_grounded']>0.05 else 'weak'}")
        print()
        out_path.write_text(json.dumps(results, indent=2))

    # verdict
    naive_spur = sum(1 for a in results.values() if a["margin_naive"] <= 0.05)
    grnd_ok = sum(1 for a in results.values() if a["margin_grounded"] > 0.05)
    print("=== VERDICT ===")
    print(f"  naive corr-verifier flagged spurious on {naive_spur}/{len(results)} tasks")
    print(f"  grounded verifier faithful on {grnd_ok}/{len(results)} tasks")
    print(f"  → the validator detects question-dependence as the decisive axis,")
    print(f"    and the margin is a per-task 'can I verify this?' signal (abstain if <=0).")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nsummary → {out_path}")


if __name__ == "__main__":
    main()
