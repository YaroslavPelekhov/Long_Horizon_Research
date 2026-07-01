"""NewtonBench via the SAME conservation-law selection shell as DiscoveryBench.

Tests whether the universal conservation mechanism transfers. Model authors K candidate
laws (closed forms). Shell fits the constant, then measures CONSERVATION = the held-out
predictive error is LOW and STABLE across resamples of the observation set (a correct law
predicts consistently regardless of which observations you fit on). Selects the law with
the lowest stable error; abstains if none is stable-low. Official evaluate_law scores it.

Identical principle to run_db_invariants (truth invariant under data resample). Here the
honest hypothesis under test: conservation-SELECTION cannot rescue NB if the candidate POOL
is poisoned by physics priors (counterfactual laws break F=Gm1m2/r^2 etc.). Bottleneck =
generation, not selection.

  python -m mars.runners.run_nb_conservation --run_id nbcons_v1 --model openai/gpt-4o-mini
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
from mars.runners.run_nb_selfverify import (_params, collect, author_laws,
                                            _compile_law, _fit_const, _rmsle)

WEAK = "openai/gpt-4o-mini"
MODULES = ["m0_gravity", "m1_coulomb_force", "m9_hooke_law",
           "m4_snell_law", "m5_radioactive_decay", "m7_malus_law"]


def conservation_rmsle(fn, kfit, rows, n_resamples=12, seed=0):
    """Conservation analog for NB: predictive error LOW and STABLE across observation
    resamples. Returns (mean_rmsle, std_rmsle). A correct law -> low mean, low std."""
    rng = np.random.RandomState(seed)
    n = len(rows)
    errs = []
    for i in range(n_resamples):
        idx = rng.randint(0, n, size=n) if i % 2 == 0 else rng.permutation(n)[: max(4, int(n * 0.6))]
        sub = [rows[j] for j in idx]
        errs.append(_rmsle(fn, kfit, sub))
    errs = np.array(errs)
    return float(np.mean(errs)), float(np.std(errs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="nbcons_v1")
    ap.add_argument("--model", default=WEAK)
    ap.add_argument("--modules", default=",".join(MODULES))
    ap.add_argument("--difficulty", default="easy")
    ap.add_argument("--mean_thresh", type=float, default=0.15)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out_dir = _PROJ / "lmw" / "nb_conservation" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit("exists; --overwrite")

    client = make_openai_client()
    results = {}
    sa_list = []
    n_ans = n_abs = 0
    print(f"=== NewtonBench — conservation-law selection (same shell as DB) — {args.model} ===\n")
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
            mean_rm, std_rm = conservation_rmsle(fn, kfit, ho)
            scored.append({"code": code, "kfit": kfit, "mean": round(mean_rm, 4),
                           "std": round(std_rm, 4), "score": round(mean_rm + std_rm, 4)})
        if not scored:
            results[mod_name] = {"status": "ABSTAIN", "why": "no law compiled"}; n_abs += 1
            print(f"  {mod_name:20} ABSTAIN (no law)"); continue

        # conserved = low AND stable predictive error
        conserved = [c for c in scored if c["mean"] < args.mean_thresh]
        if not conserved:
            best = min(c["mean"] for c in scored)
            results[mod_name] = {"status": "ABSTAIN", "best_mean_rmsle": round(best, 3)}; n_abs += 1
            print(f"  {mod_name:20} ABSTAIN (nothing conserved; best mean rmsle={best:.2f})"); continue

        best = min(conserved, key=lambda c: c["score"])
        body = best["code"].split("return", 1)[1].strip()
        dict_args = "{" + ", ".join(f"{p!r}: {p}" for p in params) + "}"
        sub = f"{best['code']}\n\n{sig}\n    return {best['kfit']!r} * law({dict_args})\n"
        try:
            ev = module.evaluate_law(sub, param_description=getattr(module, "PARAM_DESCRIPTION", ""),
                                     difficulty=args.difficulty, law_version="v0", judge_model_name="gpt41")
            sa = float(ev.get("exact_accuracy", 0.0))
        except Exception:
            sa = 0.0
        sa_list.append(sa); n_ans += 1
        results[mod_name] = {"status": "ANSWER", "SA": sa, "mean_rmsle": best["mean"],
                             "std_rmsle": best["std"], "law": body[:70]}
        print(f"  {mod_name:20} SA={sa:.2f} cons(mean={best['mean']:.3f},std={best['std']:.3f}) | {body[:42]}")
        out_path.write_text(json.dumps(results, indent=2, default=str))

    n = n_ans + n_abs
    mean_ans = sum(sa_list) / len(sa_list) if sa_list else 0.0
    mean_all = sum(sa_list) / n if n else 0.0
    print(f"\n=== RESULT ({args.difficulty}) ===")
    print(f"  answered {n_ans}/{n}  abstained {n_abs}/{n}")
    print(f"  SA on answered = {mean_ans:.2f}  |  SA over all = {mean_all:.2f}")
    print(f"  compare: active-probing (uses run_experiment API) got SA_all=0.83 | self-debug 0.17")
    print(f"  hypothesis: conservation-SELECTION can't rescue NB (pool poisoned by priors)")
    out_path.write_text(json.dumps({"model": args.model, "answered": n_ans, "abstained": n_abs,
                                    "SA_answered": round(mean_ans, 3), "SA_all": round(mean_all, 3),
                                    "results": results}, indent=2, default=str))
    print(f"\nsummary → {out_path}")


if __name__ == "__main__":
    main()
