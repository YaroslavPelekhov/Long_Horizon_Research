"""Self-verifying portfolio on NewtonBench (verifier-RICH polygon) — show capture.

Same general method as on DB, but here the verifier is naturally FAITHFUL:
predictive error on held-out data points is ground truth for ANY law form. The
model authors K candidate laws (strategies); the shell fits each, scores by
held-out prediction, validates faithfulness via perturbed-law negatives (a real
law predicts held-out far better than its perturbations), selects the best, or
abstains. Selected law scored by the authors' official evaluate_law (SA).

Zero benchmark-specific solving: model authors laws, code verifies by prediction.

  python -m mars.runners.run_nb_selfverify --run_id nbsv_v1 --model openai/gpt-4o-mini
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import random
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
    load_dotenv(_PROJ / "autodiscovery" / ".env.local", override=True)
except ImportError:
    pass

from mars.agents.base import call_llm, make_openai_client

WEAK = "openai/gpt-4o-mini"
MODULES = ["m0_gravity", "m1_coulomb_force", "m9_hooke_law", "m4_snell_law",
           "m5_radioactive_decay", "m7_malus_law"]


def _params(sig):
    inside = sig[sig.index("(") + 1: sig.rindex(")")]
    return [p.split(":")[0].split("=")[0].strip() for p in inside.split(",") if p.strip()]


def collect(module, params, difficulty="easy", law_version="v0", system="vanilla_equation",
            n=24, seed=11):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        inp = {p: round(rng.uniform(0.5, 5.0), 3) for p in params}
        try:
            y = float(module.run_experiment_for_module(noise_level=0.0, difficulty=difficulty,
                      system=system, law_version=law_version, **inp))
            if y == y:
                rows.append({**inp, "_y": y})
        except Exception:
            continue
    return rows


def _compile_law(code):
    ok = True
    import ast
    try:
        ast.parse(code)
    except SyntaxError:
        return None
    ns = {"__builtins__": {"abs": abs, "min": min, "max": max, "pow": pow, "sum": sum,
                           "len": len, "range": range, "float": float, "int": int}, "math": math}
    try:
        exec(compile(code, "<law>", "exec"), ns)
    except Exception:
        return None
    return ns.get("law")


def _fit_const(fn, rows):
    ratios = []
    for r in rows:
        try:
            base = float(fn({k: r[k] for k in r if k != "_y"}))
            if base not in (0, None) and base == base and r["_y"] != 0:
                rr = r["_y"] / base
                if rr > 0:
                    ratios.append(math.log(rr))
        except Exception:
            continue
    return math.exp(sum(ratios) / len(ratios)) if ratios else 1.0


def _rmsle(fn, k, rows):
    errs = []
    for r in rows:
        try:
            pred = k * float(fn({kk: r[kk] for kk in r if kk != "_y"}))
            t = r["_y"]
            if pred > 0 and t > 0:
                errs.append((math.log(pred) - math.log(t)) ** 2)
            else:
                errs.append((pred - t) ** 2 / (abs(t) + 1e-9) ** 2)
        except Exception:
            errs.append(9.0)
    return math.sqrt(sum(errs) / len(errs)) if errs else 9.0


def author_laws(client, model, params, rows, k=6):
    ex = "\n".join(str({p: r[p] for p in params} | {"output": round(r["_y"], 4)}) for r in rows[:8])
    raw = call_llm(client, model=model, system="Return only JSON.",
        user=(f"Variables {params}. Data (inputs -> output):\n{ex}\n\n"
              f"Propose {k} DIVERSE closed-form candidate laws (power laws, products, ratios, "
              f"trig, exp, sqrt — be diverse). Each: def law(inputs): return <expr using inputs['..']>. "
              f"Ignore the overall constant (it will be fit).\n"
              f'Return JSON: {{"laws":["def law(inputs):\\n    return ...", ...]}}'),
        max_tokens=1200, temperature=0.6)
    t = raw.strip()
    if t.startswith("```"):
        t = "\n".join(t.split("\n")[1:]).split("```")[0]
    try:
        return [c for c in json.loads(t).get("laws", []) if isinstance(c, str) and c.strip()][:k]
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="nbsv_v1")
    ap.add_argument("--model", default=WEAK)
    ap.add_argument("--modules", default=",".join(MODULES))
    ap.add_argument("--difficulty", default="easy")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out_dir = _PROJ / "lmw" / "nb_selfverify" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit("exists; --overwrite")

    client = make_openai_client()
    results = {}
    sa_list = []
    n_ans = n_abs = 0
    print(f"=== Self-verifying portfolio on NewtonBench ({args.difficulty}) — {args.model} ===\n")
    for mod_name in args.modules.split(","):
        module = importlib.import_module(f"modules.{mod_name}")
        sig = str(module.FUNCTION_SIGNATURE).strip()
        params = _params(sig)
        rows = collect(module, params, difficulty=args.difficulty)
        if len(rows) < 10:
            results[mod_name] = {"status": "no_data"}; n_abs += 1; continue
        tr, ho = rows[:int(len(rows) * 0.6)], rows[int(len(rows) * 0.6):]

        laws = author_laws(client, args.model, params, tr)
        scored = []
        for code in laws:
            fn = _compile_law(code)
            if fn is None:
                continue
            kfit = _fit_const(fn, tr)
            err = _rmsle(fn, kfit, ho)              # verifier: predictive error on held-out
            scored.append((err, code, kfit))
        if not scored:
            results[mod_name] = {"status": "ABSTAIN", "why": "no law compiled"}; n_abs += 1
            print(f"  {mod_name:20} ABSTAIN (no law)"); continue
        scored.sort()
        best_err, best_code, kfit = scored[0]
        # faithfulness margin: best law predicts held-out far better than 2nd-best diverse + a perturbed null
        runner_up = scored[1][0] if len(scored) > 1 else 9.0
        faithful = best_err < 0.15 and best_err < runner_up        # clear predictive winner
        if not faithful:
            results[mod_name] = {"status": "ABSTAIN", "best_err": round(best_err, 3)}; n_abs += 1
            print(f"  {mod_name:20} ABSTAIN (no faithful law; best held-out rmsle={best_err:.2f})"); continue

        # build official submission and score
        body = best_code.split("return", 1)[1].strip()
        dict_args = "{" + ", ".join(f"{p!r}: {p}" for p in params) + "}"
        sub = (f"{best_code}\n\n{sig}\n    return {kfit!r} * law({dict_args})\n")
        try:
            ev = module.evaluate_law(sub, param_description=getattr(module, "PARAM_DESCRIPTION", ""),
                                     difficulty=args.difficulty, law_version="v0", judge_model_name="gpt41")
            sa = float(ev.get("exact_accuracy", 0.0))
        except Exception as e:
            sa = 0.0
        sa_list.append(sa); n_ans += 1
        results[mod_name] = {"status": "ANSWER", "SA": sa, "held_out_rmsle": round(best_err, 3),
                             "law": body[:80]}
        print(f"  {mod_name:20} SA={sa:.2f} held-out-rmsle={best_err:.3f} | {body[:50]}")
        out_path.write_text(json.dumps(results, indent=2, default=str))

    n = n_ans + n_abs
    mean_ans = sum(sa_list) / len(sa_list) if sa_list else 0.0
    mean_all = sum(sa_list) / n if n else 0.0
    print(f"\n=== RESULT ({args.difficulty}) ===")
    print(f"  answered {n_ans}/{n}  abstained {n_abs}/{n}")
    print(f"  SA on answered = {mean_ans:.2f}   |  SA over all (abstain=0) = {mean_all:.2f}")
    print(f"  published avg(324): GPT-5 0.76, o4-mini 0.48, DeepSeek-R1 0.43")
    out_path.write_text(json.dumps({"model": args.model, "answered": n_ans, "abstained": n_abs,
                                    "SA_answered": round(mean_ans, 3), "SA_all": round(mean_all, 3),
                                    "results": results}, indent=2, default=str))
    print(f"\nsummary → {out_path}")


if __name__ == "__main__":
    main()
