"""NewtonBench: model-authored fitter with an EXECUTION SELF-DEBUG loop.

The reliability fix: one-shot model code is brittle. Here the model writes a
fitter, the shell RUNS it and returns the actual error OR the held-out rmsle, the
model REVISES, and this repeats until the code runs AND predicts well (low rmsle)
or the budget is spent (then abstain). General (self-debug is universal); the
predictive held-out error is the faithful verifier; the authors' evaluate_law
scores the result (SA).

  python -m mars.runners.run_nb_selfdebug --run_id nbsd_v1 --model openai/gpt-4o-mini
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import traceback
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
    load_dotenv(_PROJ / "autodiscovery" / ".env.local", override=True)
except ImportError:
    pass

from mars.agents.base import call_llm, make_openai_client
from mars.runners.run_nb_selfverify import _params, collect
from mars.runners.run_nb_fitters import _exec, _law_from_source, _rmsle

WEAK = "openai/gpt-4o-mini"
MODULES = ["m0_gravity", "m1_coulomb_force", "m9_hooke_law", "m4_snell_law",
           "m5_radioactive_decay", "m7_malus_law"]


def run_fitter(code, train, holdout, params, entry):
    """Execute a fitter; return (rmsle, law_src, error_str). error_str='' on success."""
    ffn = _exec(code, "fit")
    if ffn is None:
        return 9.9, None, "fitter code did not compile (syntax / unsafe import)"
    try:
        src = ffn([{**{p: r[p] for p in params}, "y": r["_y"]} for r in train], list(params))
    except Exception:
        return 9.9, None, f"fit(rows, params) raised:\n{traceback.format_exc(limit=2)}"
    if not isinstance(src, str):
        return 9.9, None, f"fit returned {type(src).__name__}, not a law SOURCE string"
    if entry not in src:
        return 9.9, None, f"returned source does not define `{entry}` with the required signature"
    law = _law_from_source(src, entry)
    if law is None:
        return 9.9, src, "the returned law source did not compile/exec"
    try:
        rm = _rmsle(law, holdout, params)
    except Exception:
        return 9.9, src, f"law evaluation raised:\n{traceback.format_exc(limit=2)}"
    return rm, src, ""


def self_debug_fit(client, model, params, train, holdout, sig, entry, max_rounds=4):
    ex = "\n".join(str({p: r[p] for p in params} | {"y": round(r["_y"], 5)}) for r in train[:8])
    base = (f"Variables {params}. Sample data (inputs and y):\n{ex}\n\n"
            f"Write `def fit(rows, params):` that FITS a law to the data with numpy (rows: list of "
            f"dicts, each has the params and target under key 'y') and RETURNS a Python SOURCE "
            f"STRING defining the law with EXACTLY this signature and FITTED numeric constants:\n"
            f"  {sig}\n"
            f"For power laws use log-linear lstsq on log|x| and SNAP exponents to nearest 0.5. "
            f"The returned source must use ONLY the param names and plain python/math (no numpy).\n")
    history = ""
    best = (9.9, None)
    prev_code = ""
    for rnd in range(max_rounds):
        prompt = base + (f"\nYour previous attempt:\n{prev_code}\n\nFEEDBACK: {history}\n"
                         f"Revise the fitter to fix this." if history else
                         f'\nReturn JSON: {{"code":"def fit(rows, params):\\n    ..."}}')
        if history:
            prompt += '\nReturn JSON: {"code":"def fit(rows, params):\\n    ..."}'
        raw = call_llm(client, model=model, system="Return only JSON.", user=prompt,
                       max_tokens=1400, temperature=0.3 if history else 0.5)
        t = raw.strip()
        if t.startswith("```"):
            t = "\n".join(t.split("\n")[1:]).split("```")[0]
        try:
            code = json.loads(t).get("code", "")
        except Exception:
            i, j = t.find("{"), t.rfind("}")
            try: code = json.loads(t[i:j+1]).get("code", "")
            except Exception: code = ""
        prev_code = code or prev_code
        rm, src, err = run_fitter(code, train, holdout, params, entry)
        status = err if err else f"ran OK, held-out rmsle={rm:.3f}"
        print(f"      round {rnd}: {status[:90]}", flush=True)
        if not err and rm < best[0]:
            best = (rm, src)
        if not err and rm < 0.05:
            break
        history = (err if err else
                   f"the code ran but held-out rmsle={rm:.3f} is too high (the law mispredicts). "
                   f"Try snapping exponents to nearest 0.5, or a different functional family "
                   f"(exp/log/trig/ratio). Use only param names + math in the returned source.")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="nbsd_v1")
    ap.add_argument("--model", default=WEAK)
    ap.add_argument("--modules", default=",".join(MODULES))
    ap.add_argument("--difficulty", default="easy")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out_dir = _PROJ / "lmw" / "nb_selfdebug" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit("exists; --overwrite")

    client = make_openai_client()
    results = {}; sa_list = []; n_ans = n_abs = 0
    print(f"=== NewtonBench self-debug fitter loop — {args.model} ===\n")
    for mod_name in args.modules.split(","):
        module = importlib.import_module(f"modules.{mod_name}")
        sig = str(module.FUNCTION_SIGNATURE).strip()
        entry = sig[4:sig.index("(")].strip()
        params = _params(sig)
        rows = collect(module, params, difficulty=args.difficulty)
        if len(rows) < 10:
            results[mod_name] = {"status": "no_data"}; n_abs += 1; continue
        tr, ho = rows[:int(len(rows) * 0.6)], rows[int(len(rows) * 0.6):]
        print(f"  {mod_name}:")
        rm, src = self_debug_fit(client, args.model, params, tr, ho, sig, entry, args.rounds)
        if src is None or rm > 0.5:
            results[mod_name] = {"status": "ABSTAIN", "best_rmsle": round(rm, 3)}; n_abs += 1
            print(f"      → ABSTAIN (best held-out rmsle={rm:.3f})"); continue
        try:
            ev = module.evaluate_law(src, param_description=getattr(module, "PARAM_DESCRIPTION", ""),
                                     difficulty=args.difficulty, law_version="v0", judge_model_name="gpt41")
            sa = float(ev.get("exact_accuracy", 0.0))
        except Exception:
            sa = 0.0
        sa_list.append(sa); n_ans += 1
        results[mod_name] = {"status": "ANSWER", "SA": sa, "held_out_rmsle": round(rm, 3),
                             "law": src.split("return", 1)[-1].strip()[:70]}
        print(f"      → SA={sa:.2f} rmsle={rm:.3f} | {src.split('return',1)[-1].strip()[:50]}")
        out_path.write_text(json.dumps(results, indent=2, default=str))

    n = n_ans + n_abs
    mean_ans = sum(sa_list) / len(sa_list) if sa_list else 0.0
    mean_all = sum(sa_list) / n if n else 0.0
    print(f"\n=== RESULT ({args.difficulty}) ===")
    print(f"  answered {n_ans}/{n}  abstained {n_abs}/{n}")
    print(f"  SA on answered = {mean_ans:.2f}  |  SA over all = {mean_all:.2f}")
    print(f"  specialized residual_kernel got gravity/coulomb/hooke = 1.0")
    out_path.write_text(json.dumps({"model": args.model, "answered": n_ans, "abstained": n_abs,
                                    "SA_answered": round(mean_ans, 3), "SA_all": round(mean_all, 3),
                                    "results": results}, indent=2, default=str))
    print(f"\nsummary → {out_path}")


if __name__ == "__main__":
    main()
