"""SELF-WRITING ADAPTER — universalize the architecture by removing the human from the loop.

The engine (conservation) and meta-loop (gold-free select + evolve) stay task-agnostic. The
task-specific part — HOW to discover a law from observations — is AUTHORED BY THE MODEL from a
GENERIC interface (observations = records mapping numeric inputs to a numeric output). No physics
hints, no injected basis. The model writes a discovery ALGORITHM `solve(rows, params) -> law_src`;
the shell runs it; the meta-loop keeps the solver with the best GOLD-FREE held-out fidelity across
labeled probes, evolves it via gold-free feedback, and deploys it to held-out modules.

If one model-authored solver, selected and evolved purely gold-free, transfers to held-out modules
and scores SA there, the architecture universalized itself: no per-benchmark human code, only a
generic interface + the model + the conservation/meta shell.

  python -m mars.runners.run_self_adapter_nb --run_id sa_v1 --model openai/gpt-4o-mini
"""
from __future__ import annotations

import argparse
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

# Broad-but-safe sandbox for MODEL-AUTHORED discovery algorithms: the model may use ANY standard
# scientific library it chooses (numpy/scipy/sklearn/pandas/...) — that is the model's own method
# choice, not an injected basis — while os/sys/subprocess/open stay blocked.
_ALLOWED_ROOTS = {"numpy", "scipy", "sklearn", "pandas", "math", "statistics", "collections",
                  "itertools", "functools", "operator", "random", "warnings"}


def _exec_analysis(code):
    import ast, builtins as _b
    try:
        ast.parse(code)
    except SyntaxError:
        return None

    def _imp(name, *a, **k):
        if name.split(".")[0] in _ALLOWED_ROOTS:
            return __import__(name, *a, **k)
        raise ImportError(f"blocked import: {name}")
    safe = ("abs min max pow sum len range float int str bool list dict set tuple frozenset "
            "enumerate zip map filter any all sorted reversed round divmod isinstance getattr "
            "hasattr print type repr").split()
    bd = {n: getattr(_b, n) for n in safe if hasattr(_b, n)}
    bd["__import__"] = _imp
    ns = {"__builtins__": bd}
    try:
        exec(compile(code, "<solver>", "exec"), ns)
    except Exception:
        return None
    return ns.get("solve") or ns.get("analyze")

PROBES = ["m0_gravity", "m1_coulomb_force", "m9_hooke_law"]
HOLDOUT = ["m5_radioactive_decay", "m7_malus_law", "m8_sound_speed"]

# GENERIC interface description — no physics, no basis. Applies to ANY numeric inputs->output data.
_IFACE = (
    "You are given `rows`: a list of records, each a dict mapping several numeric INPUT variables "
    "(names in `params`) to a numeric target under key 'y'. Nothing about the domain is known.")

_CONTRACT = (
    "Write `def solve(rows, params):` — a GENERAL procedure that DISCOVERS a closed-form law y = "
    "f(inputs) from the data and RETURNS Python SOURCE for `def discovered_law(<params>): return "
    "<expr>` with numeric constants FITTED from the data. It must generalize (predict unseen rows), "
    "so validate internally on a held-out split. Be general: consider multiple functional families "
    "and fit their parameters from the data; do not hard-code a single guess. Use only numpy/math. "
    "Return the law source as a string.")


def author_solvers(client, model, params, sample_rows, k=4, feedback=None, prev=None):
    ex = "\n".join(str({p: round(r[p], 4) for p in params} | {"y": round(r["_y"], 5)}) for r in sample_rows[:8])
    if feedback:
        prompt = (f"{_IFACE}\nExample rows (params={params}):\n{ex}\n\n"
                  f"Your previous discovery procedure:\n{prev}\n\nGOLD-FREE FEEDBACK — per-dataset "
                  f"held-out prediction error (9.9 = the procedure crashed or produced no valid law; "
                  f"lower is better): {feedback}\n\n"
                  f"REVISE the procedure to lower held-out error on ALL datasets and generalize. Make "
                  f"it robust (handle crashes), and broaden the functional families it fits (do not "
                  f"rely on one family); fit any exponents/constants FROM the data.\n{_CONTRACT}\n"
                  f'Return JSON: {{"solvers":["def solve(rows, params):\\n    import numpy as np\\n    ..."]}}')
        kk = max(1, k // 2)
    else:
        prompt = (f"{_IFACE}\nExample rows (params={params}):\n{ex}\n\n{_CONTRACT}\n"
                  f"Propose {k} DIVERSE such procedures (different discovery strategies).\n"
                  f'Return JSON: {{"solvers":["def solve(rows, params):\\n    import numpy as np\\n    ..."]}}')
        kk = k
    raw = call_llm(client, model=model, system="Return only JSON.", user=prompt,
                   max_tokens=2200, temperature=0.7 if not feedback else 0.5)
    t = raw.strip()
    if t.startswith("```"):
        t = "\n".join(t.split("\n")[1:]).split("```")[0]
    try:
        arr = json.loads(t).get("solvers", [])
    except Exception:
        i, j = t.find("{"), t.rfind("}")
        try: arr = json.loads(t[i:j+1]).get("solvers", [])
        except Exception: arr = []
    return [c for c in arr if isinstance(c, str) and c.strip()][:kk]


def run_solver(code, module, params, entry, difficulty, n=48):
    """Gold-free evaluation: run the authored solver on a module's data, return held-out rmsle."""
    fn = _exec_analysis(code)
    if fn is None:
        return 9.9, None
    rows = collect(module, params, difficulty=difficulty, n=n)
    if len(rows) < 12:
        return 9.9, None
    tr, ho = rows[: int(len(rows) * 0.6)], rows[int(len(rows) * 0.6):]
    try:
        src = fn([{**{p: r[p] for p in params}, "y": r["_y"]} for r in tr], list(params))
    except Exception:
        return 9.9, None
    if not isinstance(src, str) or entry not in src:
        return 9.9, None
    law = _law_from_source(src, entry)
    if law is None:
        return 9.9, src
    try:
        return _rmsle(law, ho, params), src
    except Exception:
        return 9.9, src


def eval_solver_on(client, model, code, modules, difficulty, want_sa=False):
    """Return (mean_held_out_rmsle [gold-free], mean_SA [gold, validation only], per_module_rmsle)."""
    rms, sas, per = [], [], {}
    for mod_name in modules:
        module = importlib.import_module(f"modules.{mod_name}")
        sig = str(module.FUNCTION_SIGNATURE).strip()
        entry = sig[4:sig.index("(")].strip()
        params = _params(sig)
        rm, src = run_solver(code, module, params, entry, difficulty)
        rms.append(rm); per[mod_name] = round(rm, 3)
        if want_sa and src is not None and rm < 0.05:
            try:
                ev = module.evaluate_law(src, param_description=getattr(module, "PARAM_DESCRIPTION", ""),
                                         difficulty=difficulty, law_version="v0", judge_model_name="gpt41")
                sas.append(float(ev.get("exact_accuracy", 0.0)))
            except Exception:
                sas.append(0.0)
        elif want_sa:
            sas.append(0.0)
    return float(np.mean(rms)), (float(np.mean(sas)) if sas else 0.0), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="sa_v1")
    ap.add_argument("--model", default="openai/gpt-4o-mini")
    ap.add_argument("--difficulty", default="easy")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out_dir = _PROJ / "lmw" / "self_adapter_nb" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit("exists; --overwrite")

    client = make_openai_client()
    # a generic sample of rows from one probe, only to show the model the data shape
    m0 = importlib.import_module(f"modules.{PROBES[0]}")
    p0 = _params(str(m0.FUNCTION_SIGNATURE).strip())
    sample = collect(m0, p0, difficulty=args.difficulty, n=16)

    print(f"=== SELF-WRITING ADAPTER on NewtonBench — {args.model} ===")
    print(f"    model authors the discovery ALGORITHM from a generic interface; meta-loop selects/")
    print(f"    evolves GOLD-FREE on probes {PROBES}; deploy to holdout {HOLDOUT}\n")

    # round 0: author diverse solvers, score gold-free on probes
    pool = []
    for code in author_solvers(client, args.model, p0, sample, k=6):
        rm, _sa, per = eval_solver_on(client, args.model, code, PROBES, args.difficulty)
        pool.append({"code": code, "probe_rmsle": rm, "per": per})
        print(f"    [R0] authored solver: probe mean held-out rmsle = {rm:.4f}  per={per}", flush=True)
    if not pool:
        print("  no solver authored"); return
    pool.sort(key=lambda c: c["probe_rmsle"])
    best = pool[0]

    # evolution rounds: per-module GOLD-FREE feedback -> revise best -> keep if improved
    trace = [{"round": 0, "best_probe_rmsle": round(best["probe_rmsle"], 4)}]
    for rnd in range(1, args.rounds + 1):
        fb = json.dumps(best["per"])
        revs = author_solvers(client, args.model, p0, sample, k=5, feedback=fb, prev=best["code"])
        improved = False
        for code in revs:
            rm, _sa, per = eval_solver_on(client, args.model, code, PROBES, args.difficulty)
            print(f"    [R{rnd}] revised solver: probe mean held-out rmsle = {rm:.4f}  per={per}", flush=True)
            if rm < best["probe_rmsle"]:
                best = {"code": code, "probe_rmsle": rm, "per": per}; improved = True
        trace.append({"round": rnd, "best_probe_rmsle": round(best["probe_rmsle"], 4), "improved": improved})

    # DEPLOY the gold-free-selected solver on held-out modules; NOW look at gold SA
    print(f"\n  deploying gold-free-selected solver (probe rmsle={best['probe_rmsle']:.4f}) to holdout...")
    ho_rmsle, ho_sa, ho_per = eval_solver_on(client, args.model, best["code"], HOLDOUT, args.difficulty, want_sa=True)
    pr_rmsle, pr_sa, pr_per = eval_solver_on(client, args.model, best["code"], PROBES, args.difficulty, want_sa=True)

    print(f"\n=== RESULT ({args.difficulty}) ===")
    print(f"  probes   : held-out rmsle={pr_rmsle:.4f}  SA={pr_sa:.2f}")
    print(f"  HOLDOUT  : held-out rmsle={ho_rmsle:.4f}  SA={ho_sa:.2f}   <- transfer of a self-authored solver")
    print(f"  (no human per-benchmark code; solver authored + evolved GOLD-FREE; SA seen only at the end)")
    out_path.write_text(json.dumps({"difficulty": args.difficulty, "evolution": trace,
                                    "probe_rmsle": pr_rmsle, "probe_SA": pr_sa,
                                    "holdout_rmsle": ho_rmsle, "holdout_SA": ho_sa,
                                    "best_solver": best["code"]}, indent=2, default=str))
    print(f"\nsummary → {out_path}")


if __name__ == "__main__":
    main()
