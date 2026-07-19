"""NewtonBench: model authors FITTING PROCEDURES (symbolic regression as code),
not closed-form guesses. The general method writes its own specialized fitter per
task and stays universal; predictive held-out error verifies; best is submitted.

Each strategy: def fit(rows, params) -> str  (fits a law family to the data with
numpy, returns the discovered law SOURCE matching the official signature, with
fitted numeric constants baked in). We execute it, verify by held-out prediction,
select the lowest-error law, and score with the authors' evaluate_law (SA).

  python -m mars.runners.run_nb_fitters --run_id nbf_v1 --model openai/gpt-4o-mini
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

WEAK = "openai/gpt-4o-mini"
MODULES = ["m0_gravity", "m1_coulomb_force", "m9_hooke_law", "m4_snell_law",
           "m5_radioactive_decay", "m7_malus_law"]


def _exec(code, fn_name):
    """Compile model code allowing numpy/math imports (for fitting)."""
    try:
        ast.parse(code)
    except SyntaxError:
        return None
    import numpy as np
    def _imp(name, *a, **k):
        if name.split(".")[0] in ("numpy", "math", "statistics"):
            return __import__(name, *a, **k)
        raise ImportError(name)
    ns = {"__builtins__": {"abs": abs, "min": min, "max": max, "pow": pow, "sum": sum,
                           "len": len, "range": range, "float": float, "int": int,
                           "enumerate": enumerate, "zip": zip, "list": list, "dict": dict,
                           "sorted": sorted, "round": round, "__import__": _imp},
          "np": np, "math": math}
    try:
        exec(compile(code, "<fit>", "exec"), ns)
    except Exception:
        return None
    return ns.get(fn_name)


def _law_from_source(src, entry):
    import numpy as np
    def _imp(name, *a, **k):
        if name.split(".")[0] in ("numpy", "math"):
            return __import__(name, *a, **k)
        raise ImportError(name)
    ns = {"__builtins__": {"abs": abs, "min": min, "max": max, "pow": pow, "float": float,
                           "int": int, "sum": sum, "len": len, "range": range, "__import__": _imp},
          "math": math, "np": np}
    try:
        ast.parse(src); exec(compile(src, "<law>", "exec"), ns)
    except Exception:
        return None
    return ns.get(entry)


def _rmsle(fn, rows, params):
    errs = []
    for r in rows:
        try:
            pred = float(fn(**{p: r[p] for p in params}))
            t = r["_y"]
            if pred > 0 and t > 0:
                errs.append((math.log(pred) - math.log(t)) ** 2)
            else:
                errs.append(min(9.0, (pred - t) ** 2 / (abs(t) + 1e-9) ** 2))
        except Exception:
            errs.append(9.0)
    return math.sqrt(sum(errs) / len(errs)) if errs else 9.0


def author_fitters(client, model, params, rows, sig, entry, k=5):
    ex = "\n".join(str({p: r[p] for p in params} | {"y": round(r["_y"], 5)}) for r in rows[:8])
    raw = call_llm(client, model=model, system="Return only JSON.",
        user=(f"Variables {params}. Sample data (inputs and y):\n{ex}\n\n"
              f"Write {k} DIVERSE `def fit(rows, params):` procedures. Each FITS a law family to "
              f"the data with numpy (rows: list of dicts each with the params and the target under "
              f"key 'y'), then "
              f"RETURNS a Python SOURCE STRING defining the law with this EXACT signature and the "
              f"FITTED numeric constants baked in:\n  {sig}\n"
              f"Families to cover: (1) power law via log-linear lstsq on log|x| with exponents "
              f"snapped to nearest 0.5; (2) linear/polynomial; (3) ratio/product; (4) exponential "
              f"(log-linear in x); (5) trig (fit amplitude/phase). Return the law as a string like "
              f"'{entry}(...):\\n    return <expr with numbers>'.\n"
              f'Return JSON: {{"fitters":["def fit(rows, params):\\n    import numpy as np\\n    ...\\n    return law_src", ...]}}'),
        max_tokens=2200, temperature=0.6)
    t = raw.strip()
    if t.startswith("```"):
        t = "\n".join(t.split("\n")[1:]).split("```")[0]
    try:
        return [c for c in json.loads(t).get("fitters", []) if isinstance(c, str) and c.strip()][:k]
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="nbf_v1")
    ap.add_argument("--model", default=WEAK)
    ap.add_argument("--modules", default=",".join(MODULES))
    ap.add_argument("--difficulty", default="easy")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out_dir = _PROJ / "lmw" / "nb_fitters" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit("exists; --overwrite")

    client = make_openai_client()
    results = {}; sa_list = []; n_ans = n_abs = 0
    print(f"=== NewtonBench: model-authored FITTERS + predictive verify — {args.model} ===\n")
    for mod_name in args.modules.split(","):
        module = importlib.import_module(f"modules.{mod_name}")
        sig = str(module.FUNCTION_SIGNATURE).strip()
        entry = sig[4:sig.index("(")].strip()
        params = _params(sig)
        rows = collect(module, params, difficulty=args.difficulty)
        if len(rows) < 10:
            results[mod_name] = {"status": "no_data"}; n_abs += 1; continue
        tr, ho = rows[:int(len(rows) * 0.6)], rows[int(len(rows) * 0.6):]

        fitters = author_fitters(client, args.model, params, tr, sig, entry)
        scored = []
        for fcode in fitters:
            ffn = _exec(fcode, "fit")
            if ffn is None:
                continue
            try:
                src = ffn([{**{p: r[p] for p in params}, "y": r["_y"]} for r in tr], list(params))
            except Exception:
                continue
            if not isinstance(src, str) or entry not in src:
                continue
            law = _law_from_source(src, entry)
            if law is None:
                continue
            err = _rmsle(law, ho, params)             # predictive verifier on held-out
            scored.append((err, src))
        if not scored:
            results[mod_name] = {"status": "ABSTAIN", "why": "no fitter produced a valid law"}
            n_abs += 1; print(f"  {mod_name:20} ABSTAIN (no valid fitter)"); continue
        scored.sort()
        best_err, best_src = scored[0]
        if best_err > 0.5:                            # nothing fit well -> abstain honestly
            results[mod_name] = {"status": "ABSTAIN", "best_err": round(best_err, 3)}
            n_abs += 1; print(f"  {mod_name:20} ABSTAIN (best held-out rmsle={best_err:.2f})"); continue
        try:
            ev = module.evaluate_law(best_src, param_description=getattr(module, "PARAM_DESCRIPTION", ""),
                                     difficulty=args.difficulty, law_version="v0", judge_model_name="gpt41")
            sa = float(ev.get("exact_accuracy", 0.0))
        except Exception:
            sa = 0.0
        sa_list.append(sa); n_ans += 1
        results[mod_name] = {"status": "ANSWER", "SA": sa, "held_out_rmsle": round(best_err, 3),
                             "law": best_src.split("return", 1)[-1].strip()[:70]}
        print(f"  {mod_name:20} SA={sa:.2f} rmsle={best_err:.3f} | {best_src.split('return',1)[-1].strip()[:50]}")
        out_path.write_text(json.dumps(results, indent=2, default=str))

    n = n_ans + n_abs
    mean_ans = sum(sa_list) / len(sa_list) if sa_list else 0.0
    mean_all = sum(sa_list) / n if n else 0.0
    print(f"\n=== RESULT ({args.difficulty}) ===")
    print(f"  answered {n_ans}/{n}  abstained {n_abs}/{n}")
    print(f"  SA on answered = {mean_ans:.2f}   |  SA over all (abstain=0) = {mean_all:.2f}")
    print(f"  prior: residual_kernel specialized got gravity/coulomb/hooke = 1.0")
    print(f"  published avg(324): GPT-5 0.76, o4-mini 0.48, DeepSeek-R1 0.43")
    out_path.write_text(json.dumps({"model": args.model, "answered": n_ans, "abstained": n_abs,
                                    "SA_answered": round(mean_ans, 3), "SA_all": round(mean_all, 3),
                                    "results": results}, indent=2, default=str))
    print(f"\nsummary → {out_path}")


if __name__ == "__main__":
    main()
