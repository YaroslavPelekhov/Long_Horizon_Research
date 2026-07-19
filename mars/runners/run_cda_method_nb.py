"""CDA at the METHOD level — bootstrap a working discovery method by ACCUMULATING certified
primitives instead of re-authoring a monolithic solver each round.

Self-authoring a whole method fails (brittle, SA 0). CDA fixes it with drill -> certify -> bank ->
compose:
  DRILL    the model authors a SMALL focused `def fit(rows, params) -> law_src` for the probe the
           current library fits WORST (residual-driven, gold-free — it only sees data + held-out error)
  CERTIFY  run it on all probes; keep it iff it drives held-out rmsle to ~0 on SOME probe
  BANK     add certified primitives to a growing library
  COMPOSE  the solver = portfolio over banked primitives, select per task by held-out fidelity
           (the universal conservation shell does the composition; the model only writes primitives)

As the library grows to cover more law families, the composed solver's SA should CLIMB — the
method-library self-accumulates, machine-doing what the human did by hand. Gold (SA) is read only
for validation; all drilling/certification/composition is gold-free.

  python -m mars.runners.run_cda_method_nb --run_id cda_v1 --model openai/gpt-4o-mini --drills 8
"""
from __future__ import annotations

import argparse
import ast
import importlib
import json
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
    load_dotenv(_PROJ / "autodiscovery" / ".env.local", override=False)
except ImportError:
    pass

from mars.agents.base import call_llm, make_openai_client
from mars.runners.run_nb_selfverify import _params, collect
from mars.runners.run_nb_fitters import _law_from_source, _rmsle

PROBES = ["m0_gravity", "m1_coulomb_force", "m9_hooke_law"]
HOLDOUT = ["m5_radioactive_decay", "m7_malus_law", "m8_sound_speed"]
_ALLOWED = {"numpy", "scipy", "sklearn", "pandas", "math", "statistics", "collections",
            "itertools", "functools", "operator", "random", "warnings"}


def _exec_fit(code):
    import builtins as _b
    try:
        ast.parse(code)
    except SyntaxError:
        return None

    def _imp(name, *a, **k):
        if name.split(".")[0] in _ALLOWED:
            return __import__(name, *a, **k)
        raise ImportError(name)
    safe = ("abs min max pow sum len range float int str bool list dict set tuple frozenset "
            "enumerate zip map filter any all sorted reversed round divmod isinstance getattr "
            "hasattr print type repr").split()
    bd = {n: getattr(_b, n) for n in safe if hasattr(_b, n)}
    bd["__import__"] = _imp
    ns = {"__builtins__": bd}
    try:
        exec(compile(code, "<fit>", "exec"), ns)
    except Exception:
        return None
    return ns.get("fit")


def fit_law_src(prim_code, tr, params):
    fn = _exec_fit(prim_code)
    if fn is None:
        return None
    try:
        src = fn([{**{p: r[p] for p in params}, "y": r["_y"]} for r in tr], list(params))
    except Exception:
        return None
    return src if isinstance(src, str) and "discovered_law" in src else None


def eval_primitive(prim_code, module, params, entry, difficulty, n=48):
    rows = collect(module, params, difficulty=difficulty, n=n)
    if len(rows) < 12:
        return 9.9, None
    tr, ho = rows[: int(len(rows) * 0.6)], rows[int(len(rows) * 0.6):]
    src = fit_law_src(prim_code, tr, params)
    if src is None:
        return 9.9, None
    law = _law_from_source(src, entry)
    if law is None:
        return 9.9, src
    try:
        return _rmsle(law, ho, params), src
    except Exception:
        return 9.9, src


def compose_eval(banked, modules, difficulty, want_sa=False, n=48):
    """The composed solver = portfolio over banked primitives; select per module by held-out
    fidelity (conservation). Returns per-module (best_rmsle, SA, best_src)."""
    per = {}
    for mod_name in modules:
        module = importlib.import_module(f"modules.{mod_name}")
        sig = str(module.FUNCTION_SIGNATURE).strip()
        entry = sig[4:sig.index("(")].strip()
        params = _params(sig)
        best = (9.9, None)
        for prim in banked:
            rm, src = eval_primitive(prim["code"], module, params, entry, difficulty, n)
            if src is not None and rm < best[0]:
                best = (rm, src)
        sa = 0.0
        if want_sa and best[1] is not None and best[0] < 0.05:
            try:
                ev = module.evaluate_law(best[1], param_description=getattr(module, "PARAM_DESCRIPTION", ""),
                                         difficulty=difficulty, law_version="v0", judge_model_name="gpt41")
                sa = float(ev.get("exact_accuracy", 0.0))
            except Exception:
                sa = 0.0
        per[mod_name] = {"rmsle": round(best[0], 4), "SA": sa, "src": best[1]}
    return per


def author_primitive(client, model, params, worst_rows, covered_desc, feedback=None, prev=None):
    ex = "\n".join(str({p: round(r[p], 4) for p in params} | {"y": round(r["_y"], 5)}) for r in worst_rows[:10])
    if feedback:
        prompt = (
            f"Data records map numeric inputs {params} to target 'y':\n{ex}\n\n"
            f"Your previous `fit` procedure:\n{prev}\n\nIt MISPREDICTS — held-out error and the law "
            f"it produced:\n{feedback}\n\nDEBUG YOUR OWN CODE and fix it so the held-out error goes to "
            f"~0. Check: (1) a MISSING CONSTANT/INTERCEPT term in the regression (fit an intercept, do "
            f"not force through the origin); (2) if the error stays high, the FUNCTIONAL FAMILY is "
            f"probably WRONG for this data — SWITCH family (power↔exponential↔trig↔polynomial↔ratio); "
            f"(3) GENERALITY: use the `params` argument to build the law — do NOT hard-code specific "
            f"variable names, it must work for ANY variables. Emit the law with plain arithmetic/math.\n"
            f'Return JSON: {{"family":"<name>","code":"def fit(rows, params):\\n    import numpy as np\\n    ..."}}')
    else:
        prompt = (
            f"Data records map numeric inputs {params} to target 'y'. The current library of fit-"
            f"procedures does NOT fit this dataset well:\n{ex}\n\n"
            f"Already-covered families: {covered_desc or 'none yet'}.\n\n"
            f"Write ONE small, ROBUST `def fit(rows, params):` that fits a SINGLE functional family "
            f"(power / product / exponential / trig / polynomial / ratio — pick what THIS data looks "
            f"like) and returns Python SOURCE for `def discovered_law(<params>): return <expr>` with "
            f"constants FITTED from the data. For power/product laws fit log|y| ~ intercept + sum(a_i*"
            f"log|x_i|) INCLUDING an intercept term (the overall constant). CRITICAL — GENERALITY: build "
            f"the law from the `params` argument; do NOT hard-code specific variable names, the primitive "
            f"must work for ANY input variables. Keep it crash-free; use numpy.\n"
            f'Return JSON: {{"family":"<name>","code":"def fit(rows, params):\\n    import numpy as np\\n    ..."}}')
    raw = call_llm(client, model=model, system="Return only JSON.", user=prompt,
                   max_tokens=1400, temperature=0.6 if not feedback else 0.4)
    t = raw.strip()
    if t.startswith("```"):
        t = "\n".join(t.split("\n")[1:]).split("```")[0]
    try:
        o = json.loads(t)
    except Exception:
        i, j = t.find("{"), t.rfind("}")
        try: o = json.loads(t[i:j+1])
        except Exception: return None, None
    return o.get("family", "?"), o.get("code")


def worst_probe(banked, difficulty):
    """Find the probe the current library fits worst (gold-free), return its rows + name."""
    if not banked:
        m = PROBES[0]
        module = importlib.import_module(f"modules.{m}")
        params = _params(str(module.FUNCTION_SIGNATURE).strip())
        return m, collect(module, params, difficulty=difficulty, n=48), params
    per = compose_eval(banked, PROBES, difficulty)
    worst = max(PROBES, key=lambda m: per[m]["rmsle"])
    module = importlib.import_module(f"modules.{worst}")
    params = _params(str(module.FUNCTION_SIGNATURE).strip())
    return worst, collect(module, params, difficulty=difficulty, n=48), params


def main():
    global PROBES, HOLDOUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="cda_v1")
    ap.add_argument("--model", default="openai/gpt-4o-mini")
    ap.add_argument("--difficulty", default="easy")
    ap.add_argument("--drills", type=int, default=8)
    ap.add_argument("--probes", default=",".join(PROBES))
    ap.add_argument("--holdout", default=",".join(HOLDOUT))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    PROBES = args.probes.split(",")
    HOLDOUT = args.holdout.split(",")
    out_dir = _PROJ / "lmw" / "cda_method_nb" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit("exists; --overwrite")

    client = make_openai_client()
    banked, trace = [], []
    print(f"=== CDA at the METHOD level on NewtonBench — {args.model} ===")
    print(f"    drill -> certify -> bank -> compose; probes {PROBES}; holdout {HOLDOUT}\n")

    def certify(code):
        """Best held-out rmsle over probes + the (src of the) best-fit probe (for debug feedback)."""
        best_rm, best_src = 9.9, None
        for m in PROBES:
            module = importlib.import_module(f"modules.{m}")
            pr = _params(str(module.FUNCTION_SIGNATURE).strip())
            entry = str(module.FUNCTION_SIGNATURE).strip()[4:].split("(")[0].strip()
            rm, src = eval_primitive(code, module, pr, entry, args.difficulty)
            if rm < best_rm:
                best_rm, best_src = rm, src
        return best_rm, best_src

    for d in range(args.drills):
        worst_name, worst_rows, worst_params = worst_probe(banked, args.difficulty)
        covered = ", ".join(sorted({b["family"] for b in banked})) if banked else ""
        fam, code = author_primitive(client, args.model, worst_params, worst_rows, covered)
        if not code:
            print(f"  drill {d}: (no primitive authored)"); continue
        best_rm, best_src = certify(code)
        # PRIMITIVE SELF-DEBUG: if not certified, show the model its own mispredicting law + retry
        for _dbg in range(3):
            if best_rm < 0.05:
                break
            fb = f"held-out rmsle={best_rm:.3f}; law it produced: {(best_src or '')[:200]}"
            fam2, code2 = author_primitive(client, args.model, worst_params, worst_rows, covered,
                                           feedback=fb, prev=code)
            if not code2:
                break
            rm2, src2 = certify(code2)
            if rm2 < best_rm:
                code, best_rm, best_src, fam = code2, rm2, src2, (fam2 or fam)
        certified = best_rm < 0.05
        if certified:
            banked.append({"family": fam, "code": code, "cert_rmsle": round(best_rm, 4)})
        per = compose_eval(banked, PROBES, args.difficulty, want_sa=True) if banked else {}
        probe_sa = float(np.mean([per[m]["SA"] for m in PROBES])) if per else 0.0
        print(f"  drill {d}: family='{fam[:16]:16}' worst={worst_name:16} cert_rmsle={best_rm:.4f} "
              f"{'BANKED ✓' if certified else 'rejected'}  |library|={len(banked)}  composed_probe_SA={probe_sa:.2f}",
              flush=True)
        trace.append({"drill": d, "family": fam, "certified": certified, "cert_rmsle": round(best_rm, 4),
                      "lib_size": len(banked), "composed_probe_SA": round(probe_sa, 3)})
        out_path.write_text(json.dumps({"trace": trace, "library": [b["family"] for b in banked]}, indent=2))

    # DEPLOY composed solver on held-out modules
    print(f"\n  deploying composed solver (|library|={len(banked)}) to holdout...")
    ho_per = compose_eval(banked, HOLDOUT, args.difficulty, want_sa=True) if banked else {}
    pr_per = compose_eval(banked, PROBES, args.difficulty, want_sa=True) if banked else {}
    pr_sa = float(np.mean([pr_per[m]["SA"] for m in PROBES])) if pr_per else 0.0
    ho_sa = float(np.mean([ho_per[m]["SA"] for m in HOLDOUT])) if ho_per else 0.0

    print(f"\n=== RESULT ({args.difficulty}) ===")
    print(f"  library families: {[b['family'] for b in banked]}")
    print(f"  composed solver  probe SA={pr_sa:.2f}  |  HOLDOUT SA={ho_sa:.2f}")
    print(f"  (primitives model-authored + certified; composition = universal conservation shell)")
    print(f"  vs monolithic self-author (SA 0) and hand-written unified_nb (SA 1.0)")
    out_path.write_text(json.dumps({"difficulty": args.difficulty, "library": [b["family"] for b in banked],
                                    "probe_SA": round(pr_sa, 3), "holdout_SA": round(ho_sa, 3),
                                    "probe_detail": {m: {"rmsle": pr_per[m]["rmsle"], "SA": pr_per[m]["SA"]} for m in pr_per},
                                    "holdout_detail": {m: {"rmsle": ho_per[m]["rmsle"], "SA": ho_per[m]["SA"]} for m in ho_per},
                                    "trace": trace}, indent=2, default=str))
    print(f"\nsummary → {out_path}")


if __name__ == "__main__":
    main()
