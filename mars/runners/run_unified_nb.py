"""Unified data-driven discovery: model-proposed FORM HYPOTHESES + multivariate
regression on OBSERVATIONAL data + conservation selection. No controlled experiments,
no run_experiment API — just regression on whatever rows exist. Runs unchanged on any
data with inputs and a target.

Per candidate the model proposes:
  - target_transform: 'id' or 'log'
  - per variable: a transform EXPRESSION in that variable (e.g. 'log(distance)',
    'sin(2*theta)', 'log(1+sin(2*theta))'). The MODEL supplies the nonlinearity
    (no hardcoded physics library).

The shell: evaluates the transforms numerically, fits linear coefficients by lstsq in
the transformed space (this is the prior-free generation primitive — multivariate log-log
regression recovers power-law exponents from observational data), snaps power exponents to
nearest 0.25, reconstructs the law, then CONSERVATION-selects (held-out predictive error
low AND stable across resamples). Official evaluate_law scores it.

  python -m mars.runners.run_unified_nb --run_id unb_v1 --model openai/gpt-4o-mini
"""
from __future__ import annotations

import argparse
import ast
import importlib
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
_PROJ = Path(__file__).resolve().parent.parent.parent
_NB = _PROJ / "newtonbench_repo"
for _p in (_PROJ, _NB):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
try:
    from dotenv import load_dotenv
    load_dotenv(_PROJ / "autodiscovery" / ".env.local", override=True)
except ImportError:
    pass

from mars.agents.base import call_llm, make_openai_client
from mars.runners.run_nb_selfverify import _params, collect
from mars.runners.run_nb_fitters import _law_from_source, _rmsle

WEAK = "openai/gpt-4o-mini"
MODULES = ["m0_gravity", "m1_coulomb_force", "m9_hooke_law",
           "m4_snell_law", "m5_radioactive_decay", "m7_malus_law"]

_MATH_NS = {k: getattr(math, k) for k in
            ("sin", "cos", "tan", "exp", "log", "log10", "sqrt", "pi", "e", "asin", "acos", "atan")}
_MATH_NS.update({"abs": abs, "pow": pow})


def _make_feature(expr, params):
    """Compile a feature expression over ANY of the params (SINDy-style library term).
    Returns fn(row_dict)->float or None."""
    expr = str(expr).replace("^", "**")            # forgive ^ (model sometimes means power)
    try:
        ast.parse(expr, mode="eval")
    except SyntaxError:
        return None
    arglist = ", ".join(params)
    code = f"lambda {arglist}: ({expr})"
    try:
        fn = eval(code, {"__builtins__": {}, **_MATH_NS})
    except Exception:
        return None

    def safe(row, _fn=fn):
        try:
            v = float(_fn(**{p: row[p] for p in params}))
            return v if math.isfinite(v) else None
        except Exception:
            return None
    return safe


import itertools
import re

# Universal candidate grid for typed exponent/frequency holes (covers powers AND small
# integer trig frequencies). The MODEL writes a hole token; the engine MEASURES the value.
HOLE_GRID = [-3.0, -2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
_HOLE_RE = re.compile(r"\bH\d*\b")


def _holes_in(form):
    names = set()
    for e in form.get("features", []):
        names.update(_HOLE_RE.findall(str(e)))
    return sorted(names, key=lambda s: (-len(s), s))   # longer names first for safe subst


def _subst_holes(form, mapping):
    out = {"target": form.get("target", "log"), "features": []}
    for e in form.get("features", []):
        s = str(e)
        for name in sorted(mapping, key=len, reverse=True):
            s = re.sub(rf"\b{name}\b", repr(mapping[name]), s)
        out["features"].append(s)
    return out


def expand_products(form, max_feats=6):
    """SINDy-standard library expansion: augment the model's base feature atoms with their
    PAIRWISE PRODUCTS. The model supplies the atoms (log(N0), lambda, t**H, sin(H*theta), I_0);
    the shell forms interactions (lambda*t**H, I_0*sin(H*theta)) that the weak model proposes
    separately rather than multiplied. Universal (no domain knowledge), regression+conservation
    select the sparse correct combination."""
    base = [str(e) for e in form.get("features", []) if str(e).strip()]
    feats = list(base)
    for a, b in itertools.combinations(base, 2):
        feats.append(f"({a})*({b})")
    # dedup, cap to keep lstsq well-posed
    seen, out = set(), []
    for f in feats:
        if f not in seen:
            seen.add(f); out.append(f)
    return {"target": form.get("target", "log"), "features": out[:max_feats]}


def close_holes_and_fit(form, params, rows, max_holes=2, val_rows=None, entry="discovered_law"):
    """DPSR step: if the form has typed holes (exponents/frequencies), INFER their values by
    executable measurement — sweep the grid, fit coefficients on `rows`, and score each combo
    on HELD-OUT `val_rows` (the faithful measurement: avoids over-fitting the hole when the
    SINDy-expanded library has spare free features). Falls back to in-fit residual if no val."""
    holes = _holes_in(form)
    if not holes:
        return fit_form(form, params, rows)
    if len(holes) > max_holes:
        return None, False, 9e9
    best = (None, False, 9e9)
    for combo in itertools.product(HOLE_GRID, repeat=len(holes)):
        mapping = dict(zip(holes, combo))
        concrete = _subst_holes(form, mapping)
        src, varies, resid = fit_form(concrete, params, rows)
        if src is None or not varies:
            continue
        if val_rows:
            law = _law_from_source(src, entry)
            if law is None:
                continue
            # combine in-fit residual + held-out error: robust to both over-fit (held-out
            # catches it) and thin-data noise (in-fit anchors it)
            score = _rmsle(law, val_rows, params) + resid
        else:
            score = resid
        if score < best[2]:
            best = (src, varies, score)
    return best


def author_forms(client, model, params, rows, k=6):
    ex = "\n".join(str({p: round(r[p], 4) for p in params} | {"y": round(r["_y"], 5)}) for r in rows[:10])
    raw = call_llm(client, model=model, system="Return only JSON.",
        user=(f"Variables {params}. Observational data (random input combos -> y):\n{ex}\n\n"
              f"Propose {k} DIVERSE FORM HYPOTHESES for how y depends on the variables. For each, give "
              f"a target_transform ('id' or 'log') and a LIST of feature expressions (each over any of "
              f"the variables). A LINEAR fit y_transformed ~ sum(coef_i * feature_i) + c will then be "
              f"done, so choose features that LINEARIZE the suspected form:\n"
              f"  - power/product: target='log', features=['log({params[0]})', ...] -> coefs are exponents\n"
              f"  - exponential decay exp(-k*v): target='log', features=['{params[0]}', ...]\n"
              f"  - INTERACTION inside exp: target='log', features=['x*y**H'] (H = unknown power)\n"
              f"  - trig/additive: target='id' or 'log', features=['sin(H*v)','1+sin(H*v)','cos(v)']\n"
              f"TYPED HOLES: if a POWER or FREQUENCY is unknown, write a HOLE token 'H' (or H1,H2) in "
              f"its place — the engine MEASURES it from data; NEVER guess the number yourself. "
              f"E.g. unknown decay power -> 'lambda*t**H'; unknown trig frequency -> 'sin(H*theta)'.\n"
              f"Use exactly the variable names. Mix log-features, cross-variable features and holes.\n"
              f'Return JSON: {{"forms":[{{"target":"log","features":["log({params[0]})"]}}]}}'),
        max_tokens=1700, temperature=0.6)
    t = raw.strip()
    if t.startswith("```"):
        t = "\n".join(t.split("\n")[1:]).split("```")[0]
    try:
        forms = json.loads(t).get("forms", [])
    except Exception:
        i, j = t.find("{"), t.rfind("}")
        try: forms = json.loads(t[i:j+1]).get("forms", [])
        except Exception: forms = []
    return [f for f in forms if isinstance(f, dict) and f.get("features")][:k]


def fit_form(form, params, rows):
    """Linear regression over the model-proposed feature LIBRARY (SINDy-style). Returns
    (law_source, varies, train_resid). train_resid = std of the in-fit residual in the
    transformed space (used to MEASURE typed-hole values: the right exponent drives it ~0)."""
    target = form.get("target", "log")
    feats_expr = [str(e) for e in form.get("features", []) if str(e).strip()]
    if not feats_expr:
        return None, False, 9e9
    ffns = []
    for e in feats_expr:
        fn = _make_feature(e, params)
        if fn is None:
            return None, False, 9e9
        ffns.append(fn)
    X, Y = [], []
    for r in rows:
        feats = [fn(r) for fn in ffns]
        if any(f is None for f in feats):
            continue
        y = r["_y"]
        if target == "log":
            if y <= 0:
                continue
            yv = math.log(y)
        else:
            yv = y
        X.append(feats + [1.0]); Y.append(yv)
    if len(X) < len(feats_expr) + 2:
        return None, False, 9e9
    X = np.array(X); Y = np.array(Y)
    try:
        coefs, *_ = np.linalg.lstsq(X, Y, rcond=None)
    except Exception:
        return None, False, 9e9
    coef_v = coefs[:-1]; intercept = coefs[-1]
    train_resid = float(np.std(Y - X @ coefs))
    # snap power exponents — but ONLY when the raw coef is already NEAR a grid point.
    # With noiseless data the lstsq coef is exact; snapping unconditionally would destroy
    # genuine non-quarter exponents (e.g. N0^1.2, distance^-2.72 in mutated laws).
    norm = [e.replace("^", "**").replace(" ", "") for e in feats_expr]
    snapped = []
    varies = False
    for e, c in zip(norm, coef_v):
        is_power = target == "log" and e.startswith("log(") and e.endswith(")") and \
                   e[4:-1] in params
        grid = round(c * 4) / 4
        snapped.append(grid if (is_power and abs(c - grid) < 0.03) else float(c))
        if abs(c) > 1e-6:
            varies = True
    # reconstruct
    terms = []
    for e_raw, e_norm, c in zip(feats_expr, norm, snapped):
        expr = e_raw.replace("^", "**")
        if target == "log" and e_norm.startswith("log(") and e_norm.endswith(")") and e_norm[4:-1] in params:
            terms.append(f"({e_norm[4:-1]} ** {c})")     # exp(c*log x) = x^c
        elif target == "log":
            terms.append(f"math.exp({c} * ({expr}))")
        else:
            terms.append(f"({c} * ({expr}))")
    if target == "log":
        try:
            c0 = math.exp(intercept)
        except OverflowError:
            return None, False, 9e9                  # wild hole combo -> reject
        body = f"{c0!r} * " + " * ".join(terms) if terms else f"{c0!r}"
    else:
        body = f"{intercept!r} + " + " + ".join(terms) if terms else f"{intercept!r}"
    sig = "def discovered_law(" + ", ".join(params) + "):"
    # both `math.` (used by reconstruction) and bare names (from model feature exprs like sin/log)
    return f"import math\nfrom math import *\n\n{sig}\n    return {body}\n", varies, train_resid


def author_forms_residual(client, model, params, rows, best_form, best_mean, k=6):
    """Residual-guided second round (universal self-debug for forms): the best form still
    mispredicts; ask for DIFFERENT feature families. No answer injection — only the fact
    that the current form fits poorly and which form was tried."""
    ex = "\n".join(str({p: round(r[p], 4) for p in params} | {"y": round(r["_y"], 5)}) for r in rows[:10])
    raw = call_llm(client, model=model, system="Return only JSON.",
        user=(f"Variables {params}. Observational data:\n{ex}\n\n"
              f"A linear-in-features fit was tried with this form but it MISPREDICTS "
              f"(held-out rmsle={best_mean:.2f}):\n  {json.dumps(best_form)}\n\n"
              f"Propose {k} DIFFERENT form hypotheses. SWEEP the functional family and POWERS — for any "
              f"variable whose dependence is unclear, emit the SAME structure at several powers so the "
              f"fit can pick the best (e.g. for a decay term, features ['a*b**0.5'],['a*b**1.0'],"
              f"['a*b**2.0'],['a*b**3.0'] across separate forms). Also try interactions (products of "
              f"variables inside an exponential, target='log') and trig forms ('sin(H*v)',"
              f"'1+sin(H*v)','cos(v)**2'). Use TYPED HOLES 'H'/'H1' for any unknown power or "
              f"frequency — the engine measures them from data; never guess the number. Be bold.\n"
              f'Return JSON: {{"forms":[{{"target":"log","features":["log({params[0]})"]}}]}}'),
        max_tokens=1700, temperature=0.8)
    t = raw.strip()
    if t.startswith("```"):
        t = "\n".join(t.split("\n")[1:]).split("```")[0]
    try:
        forms = json.loads(t).get("forms", [])
    except Exception:
        i, j = t.find("{"), t.rfind("}")
        try: forms = json.loads(t[i:j+1]).get("forms", [])
        except Exception: forms = []
    return [f for f in forms if isinstance(f, dict) and f.get("features")][:k]


def conservation(law, rows, params, n_resamples=12, seed=0):
    rng = np.random.RandomState(seed)
    n = len(rows)
    errs = []
    for i in range(n_resamples):
        idx = rng.randint(0, n, size=n) if i % 2 == 0 else rng.permutation(n)[: max(4, int(n*0.6))]
        sub = [rows[j] for j in idx]
        errs.append(_rmsle(law, sub, params))
    errs = np.array(errs)
    return float(np.mean(errs)), float(np.std(errs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="unb_v1")
    ap.add_argument("--model", default=WEAK)
    ap.add_argument("--modules", default=",".join(MODULES))
    ap.add_argument("--difficulty", default="easy")
    ap.add_argument("--mean_thresh", type=float, default=0.1)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out_dir = _PROJ / "lmw" / "unified_nb" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit("exists; --overwrite")

    client = make_openai_client()
    results = {}; sa_list = []; n_ans = n_abs = 0
    print(f"=== NewtonBench — UNIFIED data-driven generation + conservation — {args.model} ===\n")
    for mod_name in args.modules.split(","):
        module = importlib.import_module(f"modules.{mod_name}")
        sig = str(module.FUNCTION_SIGNATURE).strip()
        entry = sig[4:sig.index("(")].strip()
        params = _params(sig)
        rows = collect(module, params, difficulty=args.difficulty, n=36)
        if len(rows) < 10:
            results[mod_name] = {"status": "no_data"}; n_abs += 1; continue
        tr, ho = rows[:int(len(rows)*0.6)], rows[int(len(rows)*0.6):]

        # split train into fit/val so typed-hole values are measured on held-out data
        tr_fit, tr_val = tr[:int(len(tr) * 0.6)], tr[int(len(tr) * 0.6):]

        def eval_forms(forms):
            out = []
            for form0 in forms:
                # form as-is AND with SINDy pairwise-product expansion; DPSR-measure holes in each
                for form in (form0, expand_products(form0)):
                    src, varies, _ = close_holes_and_fit(form, params, tr_fit, val_rows=tr_val, entry=entry)
                    if src is None or not varies:     # degenerate constant -> no real relationship
                        continue
                    law = _law_from_source(src, entry)
                    if law is None:
                        continue
                    # degenerate guard: a real law VARIES with inputs; near-constant output
                    # (e.g. a broken/saturated module) is not a discovered relationship -> skip
                    pv = []
                    for r in ho:
                        try:
                            pv.append(float(law(**{p: r[p] for p in params})))
                        except Exception:
                            pass
                    if pv:
                        mu = sum(pv) / len(pv)
                        cv = (np.std(pv) / (abs(mu) + 1e-12))
                        if cv < 0.02:
                            continue
                    mean_rm, std_rm = conservation(law, ho, params)
                    out.append({"src": src, "mean": round(mean_rm, 4), "std": round(std_rm, 4), "form": form})
            return out

        scored = eval_forms(author_forms(client, args.model, params, tr))
        conserved = [c for c in scored if c["mean"] < args.mean_thresh]
        # residual-guided self-debug round if nothing conserved yet
        if not conserved and scored:
            best0 = min(scored, key=lambda c: c["mean"])
            more = author_forms_residual(client, args.model, params, tr, best0["form"], best0["mean"])
            scored += eval_forms(more)
            conserved = [c for c in scored if c["mean"] < args.mean_thresh]

        if not scored:
            results[mod_name] = {"status": "ABSTAIN", "why": "no form fit"}; n_abs += 1
            print(f"  {mod_name:20} ABSTAIN (no form fit)"); continue
        if not conserved:
            best = min(c["mean"] for c in scored)
            results[mod_name] = {"status": "ABSTAIN", "best_mean": round(best, 3)}; n_abs += 1
            print(f"  {mod_name:20} ABSTAIN (nothing conserved; best mean rmsle={best:.3f})"); continue
        best = min(conserved, key=lambda c: c["mean"] + c["std"])
        try:
            ev = module.evaluate_law(best["src"], param_description=getattr(module, "PARAM_DESCRIPTION", ""),
                                     difficulty=args.difficulty, law_version="v0", judge_model_name="gpt41")
            sa = float(ev.get("exact_accuracy", 0.0))
        except Exception:
            sa = 0.0
        sa_list.append(sa); n_ans += 1
        law_snip = best["src"].split("return", 1)[-1].strip()[:60]
        results[mod_name] = {"status": "ANSWER", "SA": sa, "mean_rmsle": best["mean"],
                             "std_rmsle": best["std"], "law": law_snip}
        print(f"  {mod_name:20} SA={sa:.2f} cons(mean={best['mean']:.4f}) | {law_snip[:45]}")
        out_path.write_text(json.dumps(results, indent=2, default=str))

    n = n_ans + n_abs
    mean_ans = sum(sa_list) / len(sa_list) if sa_list else 0.0
    mean_all = sum(sa_list) / n if n else 0.0
    print(f"\n=== RESULT ({args.difficulty}) ===")
    print(f"  answered {n_ans}/{n}  abstained {n_abs}/{n}")
    print(f"  SA on answered = {mean_ans:.2f}  |  SA over all = {mean_all:.2f}")
    print(f"  vs: conservation-over-prior-guesses=0.00 | active-probe(run_experiment API)=0.83 | self-debug=0.17")
    out_path.write_text(json.dumps({"model": args.model, "answered": n_ans, "abstained": n_abs,
                                    "SA_answered": round(mean_ans, 3), "SA_all": round(mean_all, 3),
                                    "results": results}, indent=2, default=str))
    print(f"\nsummary → {out_path}")


if __name__ == "__main__":
    main()
