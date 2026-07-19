"""BOREDOM-driven discovery — novelty search against mode collapse (attacks Gate 1).

Our main wall is Gate 1: the weak model collapses to its familiar modes and NEVER proposes the
counterfactual form (temperature perturbs the surface, not the STRUCTURE). "Revolutions come from
boredom" formalizes as novelty search / quality-diversity: keep an ARCHIVE of already-tried forms
and PRESSURE the model to propose something STRUCTURALLY DIFFERENT it is 'bored' of repeating —
novelty first, the verifier (held-out fidelity) selects which novel forms actually work.

Falsifiable: on modules where plain generation mode-collapses (radioactive exp(-lambda*t^1.5),
malus 1+sin(2theta)), does boredom surface the correct counterfactual form that repeated sampling
misses? Compared head-to-head with a NO-boredom baseline (independent samples, same budget).

  python -m mars.runners.run_boredom_nb --run_id bore_v1 --model openai/gpt-4o-mini
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
from mars.runners.run_unified_nb import expand_products, close_holes_and_fit, conservation

MODULES = ["m5_radioactive_decay", "m7_malus_law", "m0_gravity"]


def _parse_forms(raw, k):
    t = raw.strip()
    if t.startswith("```"):
        t = "\n".join(t.split("\n")[1:]).split("```")[0]
    try:
        arr = json.loads(t).get("forms", [])
    except Exception:
        i, j = t.find("{"), t.rfind("}")
        try: arr = json.loads(t[i:j + 1]).get("forms", [])
        except Exception: arr = []
    return [f for f in arr if isinstance(f, dict) and f.get("features")][:k]


def propose(client, model, params, sample, archive=None):
    ex = "\n".join(str({p: round(r[p], 4) for p in params} | {"y": round(r["_y"], 5)}) for r in sample[:8])
    common = (f"Variables {params}. Data (inputs->y):\n{ex}\n\n"
              f"A FORM = {{'target':'log'|'id','features':[expr,...]}}; a linear fit is done in the "
              f"transformed space. Features may use a HOLE token 'H'/'H1' for an unknown power/"
              f"frequency (the engine measures it). Examples of DIVERSE families: power 'log(x)'; "
              f"exponential decay 'x' with target log; INTERACTION inside exp 'a*b**H'; trig "
              f"'sin(H*x)','1+sin(H*x)','cos(x)**2'.\n")
    if archive:
        arch = "\n".join(f"  - {a}" for a in archive[-12:])
        prompt = (common +
                  f"You have ALREADY TRIED these forms (you are BORED of them — do not repeat their "
                  f"structure):\n{arch}\n\nPropose 3 forms that are STRUCTURALLY, QUALITATIVELY "
                  f"DIFFERENT from ALL of the above — a different functional family, a different "
                  f"interaction, a transform you have not used. Prioritize NOVELTY over immediate "
                  f"plausibility; the fit is checked afterwards. Be bold and weird.\n"
                  f'Return JSON: {{"forms":[{{"target":"log","features":["..."]}}]}}')
        temp = 0.9
    else:
        prompt = (common + f"Propose 3 plausible forms for this data.\n"
                  f'Return JSON: {{"forms":[{{"target":"log","features":["..."]}}]}}')
        temp = 0.5
    raw = call_llm(client, model=model, system="Return only JSON.", user=prompt, max_tokens=900, temperature=temp)
    return _parse_forms(raw, 3)


def eval_form(form, params, entry, tr, ho):
    tr_fit, tr_val = tr[: int(len(tr) * 0.6)], tr[int(len(tr) * 0.6):]
    best = (9.9, None)
    for f in (form, expand_products(form)):
        src, varies, _ = close_holes_and_fit(f, params, tr_fit, val_rows=tr_val, entry=entry)
        if not src or not varies:
            continue
        law = _law_from_source(src, entry)
        if law is None:
            continue
        pv = []
        for r in ho:
            try: pv.append(float(law(**{p: r[p] for p in params})))
            except Exception: pass
        if pv and np.std(pv) / (abs(np.mean(pv)) + 1e-12) < 0.02:
            continue
        m, _s = conservation(law, ho, params)
        if m < best[0]:
            best = (m, src)
    return best


def run_mode(client, model, module, params, entry, rows, rounds, bored):
    tr, ho = rows[: int(len(rows) * 0.6)], rows[int(len(rows) * 0.6):]
    archive, best = [], (9.9, None)
    for r in range(rounds):
        forms = propose(client, model, params, tr, archive=(archive if bored else None))
        for form in forms:
            rm, src = eval_form(form, params, entry, tr, ho)
            archive.append(f"target={form.get('target')} features={form.get('features')} -> rmsle={rm:.3f}")
            if rm < best[0]:
                best = (rm, src)
        if best[0] < 1e-4:
            break
    return best


def score_SA(module, src, difficulty):
    try:
        ev = module.evaluate_law(src, param_description=getattr(module, "PARAM_DESCRIPTION", ""),
                                 difficulty=difficulty, law_version="v0", judge_model_name="gpt41")
        return float(ev.get("exact_accuracy", 0.0))
    except Exception:
        return 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="bore_v1")
    ap.add_argument("--model", default="openai/gpt-4o-mini")
    ap.add_argument("--difficulty", default="easy")
    ap.add_argument("--modules", default=",".join(MODULES))
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out_dir = _PROJ / "lmw" / "boredom_nb" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit("exists; --overwrite")

    client = make_openai_client()
    results = {}
    print(f"=== BOREDOM (novelty search) vs baseline on NewtonBench — {args.model} ===")
    print(f"    same budget ({args.rounds} rounds x3 forms); does novelty surface the counterfactual?\n")
    for mod_name in args.modules.split(","):
        module = importlib.import_module(f"modules.{mod_name}")
        sig = str(module.FUNCTION_SIGNATURE).strip()
        entry = sig[4:sig.index("(")].strip()
        params = _params(sig)
        rows = collect(module, params, difficulty=args.difficulty, n=48)
        b_rm, b_src = run_mode(client, args.model, module, params, entry, rows, args.rounds, bored=False)
        n_rm, n_src = run_mode(client, args.model, module, params, entry, rows, args.rounds, bored=True)
        b_sa = score_SA(module, b_src, args.difficulty) if b_src and b_rm < 0.05 else 0.0
        n_sa = score_SA(module, n_src, args.difficulty) if n_src and n_rm < 0.05 else 0.0
        results[mod_name] = {"baseline_rmsle": round(b_rm, 4), "baseline_SA": b_sa,
                             "boredom_rmsle": round(n_rm, 4), "boredom_SA": n_sa}
        flag = "BOREDOM WINS" if n_sa > b_sa else ("tie" if n_sa == b_sa else "baseline better")
        print(f"  {mod_name:20} baseline SA={b_sa:.2f}(rmsle {b_rm:.3f})  boredom SA={n_sa:.2f}"
              f"(rmsle {n_rm:.3f})  -> {flag}", flush=True)
        out_path.write_text(json.dumps(results, indent=2, default=str))

    b_all = np.mean([r["baseline_SA"] for r in results.values()])
    n_all = np.mean([r["boredom_SA"] for r in results.values()])
    print(f"\n=== RESULT ===")
    print(f"  baseline SA_all = {b_all:.2f}   boredom SA_all = {n_all:.2f}")
    print(f"  {'novelty search expands effective support past mode-collapse' if n_all > b_all else 'no boredom gain here'}")
    out_path.write_text(json.dumps({"baseline_SA_all": round(float(b_all), 3),
                                    "boredom_SA_all": round(float(n_all), 3),
                                    "results": results}, indent=2, default=str))
    print(f"\nsummary → {out_path}")


if __name__ == "__main__":
    main()
