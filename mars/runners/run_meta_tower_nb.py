"""Recursive META-TOWER on NewtonBench — the outer loop that auto-selects the method.

"docker in docker": each level runs and judges the level below on an ever-more-held-out metric.
  L0  solve a task with a METHOD (regression / prose-guess / consensus)
  L1  select the METHOD by a GOLD-FREE signal (mean held-out predictive fidelity on probes),
      then VALIDATE the pick against gold SA (does the gold-free choice == the gold-best?)
  L2  transfer: does L1's selection on probe-distribution A also win on distribution B?
      (catches meta-overfit; if the winner differs, the tower needs another rung or the
      environment's verifiable structure is exhausted)

Falsifiable: (a) does L1 auto-rediscover `regression` WITHOUT being told, using only held-out
fidelity? (b) at what level does the tower stop helping = depth = verifiable structure of NB.

  python -m mars.runners.run_meta_tower_nb --run_id mt_v1 --model openai/gpt-4o-mini
"""
from __future__ import annotations

import argparse
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

from mars.agents.base import make_openai_client
from mars.runners.run_nb_selfverify import _params, collect, author_laws, _compile_law, _fit_const
from mars.runners.run_nb_fitters import _law_from_source, _rmsle
from mars.runners.run_unified_nb import (author_forms, expand_products, close_holes_and_fit,
                                         conservation)

WEAK = "openai/gpt-4o-mini"
PROBES_A = ["m0_gravity", "m1_coulomb_force", "m9_hooke_law"]
PROBES_B = ["m5_radioactive_decay", "m7_malus_law", "m8_sound_speed"]


def _score_SA(module, src, difficulty):
    try:
        ev = module.evaluate_law(src, param_description=getattr(module, "PARAM_DESCRIPTION", ""),
                                 difficulty=difficulty, law_version="v0", judge_model_name="gpt41")
        return float(ev.get("exact_accuracy", 0.0))
    except Exception:
        return 0.0


# ---- L0 METHODS: each returns (held_out_rmsle, SA, law_src) ; rmsle is the GOLD-FREE signal ----

def method_regression(client, model, module, params, entry, sig, rows, difficulty):
    tr, ho = rows[: int(len(rows) * 0.6)], rows[int(len(rows) * 0.6):]
    tr_fit, tr_val = tr[: int(len(tr) * 0.6)], tr[int(len(tr) * 0.6):]
    scored = []
    for f0 in author_forms(client, model, params, tr):
        for f in (f0, expand_products(f0)):
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
            scored.append((m, src))
    if not scored:
        return 9.9, 0.0, None
    scored.sort()
    rmsle, src = scored[0]
    return rmsle, _score_SA(module, src, difficulty), src


def method_prose(client, model, module, params, entry, sig, rows, difficulty):
    tr, ho = rows[: int(len(rows) * 0.6)], rows[int(len(rows) * 0.6):]
    best = (9.9, None, None)
    for code in author_laws(client, model, params, tr):
        fn = _compile_law(code)
        if fn is None:
            continue
        k = _fit_const(fn, tr)
        errs = []
        for r in ho:
            try:
                pred = k * float(fn({kk: r[kk] for kk in r if kk != "_y"}))
                t = r["_y"]
                errs.append((math.log(pred) - math.log(t)) ** 2 if pred > 0 and t > 0 else 9.0)
            except Exception:
                errs.append(9.0)
        rm = math.sqrt(sum(errs) / len(errs)) if errs else 9.9
        if rm < best[0]:
            dict_args = "{" + ", ".join(f"{p!r}: {p}" for p in params) + "}"
            src = f"{code}\n\n{sig}\n    return {k!r} * law({dict_args})\n"
            best = (rm, src, code)
    if best[1] is None:
        return 9.9, 0.0, None
    return best[0], _score_SA(module, best[1], difficulty), best[1]


def method_consensus(client, model, module, params, entry, sig, rows, difficulty):
    tr, ho = rows[: int(len(rows) * 0.6)], rows[int(len(rows) * 0.6):]
    laws = author_laws(client, model, params, tr, k=6)
    fitted = []
    for code in laws:
        fn = _compile_law(code)
        if fn is None:
            continue
        k = _fit_const(fn, tr)
        sig_vec, ok = [], True
        for r in ho:
            try:
                sig_vec.append(round(k * float(fn({kk: r[kk] for kk in r if kk != "_y"})), 3))
            except Exception:
                ok = False; break
        if ok and sig_vec:
            dict_args = "{" + ", ".join(f"{p!r}: {p}" for p in params) + "}"
            src = f"{code}\n\n{sig}\n    return {k!r} * law({dict_args})\n"
            fitted.append((json.dumps(sig_vec), src, k, fn))
    if not fitted:
        return 9.9, 0.0, None
    from collections import Counter
    keys = Counter(f[0] for f in fitted)
    top_key, n = keys.most_common(1)[0]
    rep = next(f for f in fitted if f[0] == top_key)
    # held-out rmsle of the consensus pick (gold-free signal)
    errs = []
    for r in ho:
        try:
            pred = rep[2] * float(rep[3]({kk: r[kk] for kk in r if kk != "_y"})); t = r["_y"]
            errs.append((math.log(pred) - math.log(t)) ** 2 if pred > 0 and t > 0 else 9.0)
        except Exception:
            errs.append(9.0)
    rm = math.sqrt(sum(errs) / len(errs)) if errs else 9.9
    return rm, _score_SA(module, rep[1], difficulty), rep[1]


METHODS = {"regression": method_regression, "prose_guess": method_prose, "consensus": method_consensus}


def run_level0(client, model, modules, difficulty):
    """Run every method on every probe module. Returns {method: {module: (rmsle, SA)}}."""
    out = {m: {} for m in METHODS}
    for mod_name in modules:
        module = importlib.import_module(f"modules.{mod_name}")
        sig = str(module.FUNCTION_SIGNATURE).strip()
        entry = sig[4:sig.index("(")].strip()
        params = _params(sig)
        rows = collect(module, params, difficulty=difficulty, n=36)
        for mname, fn in METHODS.items():
            try:
                rm, sa, _src = fn(client, model, module, params, entry, sig, rows, difficulty)
            except Exception:
                rm, sa = 9.9, 0.0
            out[mname][mod_name] = (rm, sa)
            print(f"    L0 [{mname:11}] {mod_name:20} held_out_rmsle={rm:.4f}  SA={sa:.2f}", flush=True)
    return out


def l1_select(level0):
    """GOLD-FREE selection: pick the method with the lowest MEAN held-out rmsle across probes."""
    means_rmsle = {m: float(np.mean([v[0] for v in d.values()])) for m, d in level0.items()}
    means_sa = {m: float(np.mean([v[1] for v in d.values()])) for m, d in level0.items()}
    picked = min(means_rmsle, key=means_rmsle.get)          # gold-free
    gold_best = max(means_sa, key=means_sa.get)             # validation only
    return picked, gold_best, means_rmsle, means_sa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="mt_v1")
    ap.add_argument("--model", default=WEAK)
    ap.add_argument("--difficulty", default="easy")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out_dir = _PROJ / "lmw" / "meta_tower_nb" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit("exists; --overwrite")

    client = make_openai_client()
    print(f"=== META-TOWER on NewtonBench — {args.model} ===\n")

    print("  [L0/L1] probe distribution A:", PROBES_A)
    l0_A = run_level0(client, args.model, PROBES_A, args.difficulty)
    pick_A, gold_A, rmsle_A, sa_A = l1_select(l0_A)
    print(f"\n  L1 on A: gold-free pick = '{pick_A}' | gold-best = '{gold_A}' | "
          f"{'MATCH ✓ (auto-rediscovered)' if pick_A == gold_A else 'MISMATCH ✗'}")
    print(f"     mean held_out_rmsle: {json.dumps({k: round(v,3) for k,v in rmsle_A.items()})}")
    print(f"     mean SA (validation): {json.dumps({k: round(v,2) for k,v in sa_A.items()})}\n")

    print("  [L2] transfer to probe distribution B:", PROBES_B)
    l0_B = run_level0(client, args.model, PROBES_B, args.difficulty)
    pick_B, gold_B, rmsle_B, sa_B = l1_select(l0_B)
    transfer = pick_A == pick_B
    print(f"\n  L2 transfer: pick on A = '{pick_A}'  vs  pick on B = '{pick_B}' | "
          f"{'TRANSFERS ✓ (no meta-overfit)' if transfer else 'DIFFERS ✗ (meta-overfit / structure exhausted)'}")

    result = {
        "L1_pick_A": pick_A, "L1_gold_best_A": gold_A, "auto_rediscovered": pick_A == gold_A,
        "L1_pick_B": pick_B, "L1_gold_best_B": gold_B, "L2_transfers": transfer,
        "rmsle_A": rmsle_A, "sa_A": sa_A, "rmsle_B": rmsle_B, "sa_B": sa_B,
        "level0_A": {m: {k: list(v) for k, v in d.items()} for m, d in l0_A.items()},
        "level0_B": {m: {k: list(v) for k, v in d.items()} for m, d in l0_B.items()},
    }
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\n=== TOWER VERDICT ===")
    print(f"  L1 auto-rediscovers the method by a gold-free signal: {result['auto_rediscovered']}")
    print(f"  L2 selection transfers across distributions: {result['L2_transfers']}")
    depth = 2 if result["auto_rediscovered"] and result["L2_transfers"] else (1 if result["auto_rediscovered"] else 0)
    print(f"  => usable tower depth on NB-{args.difficulty}: {depth}")
    print(f"\nsummary → {out_path}")


if __name__ == "__main__":
    main()
