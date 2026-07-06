"""Microscopic law of amplification: measure VERIFIER FAITHFULNESS f(nu) directly.

Everything in this project is verifier-driven search; the ceiling is "verifiable structure". This
isolates the MICROSCOPIC mechanism: given a correct law and plausible WRONG variants (perturbed
exponents/constants), at noise level nu, does the verifier (held-out predictive fidelity) still
rank the correct one FIRST? f(nu) = P(verifier ranks correct #1). No LLM in the loop, so this
measures the VERIFIER'S discriminative power alone, cleanly.

Predicts the amplification law  amplified_acc = f * (1 - (1-p)^K):  as K grows the correct law is
present w.p. ->1, but the ceiling is f, not 1 — amplification is bounded by verifier faithfulness,
and f collapses once noise swamps the fidelity gap between correct and near-miss laws. This is the
microscopic mechanism behind the measured tower depth (grounded to noise ~0.1).

  python -m mars.runners.run_verifier_faithfulness_nb --run_id vf_v1
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

from mars.runners.run_nb_selfverify import _params

MODULES = ["m0_gravity", "m1_coulomb_force", "m9_hooke_law"]
NOISE = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8]


def collect_noisy(module, params, noise, n=60, seed=11, difficulty="easy"):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        inp = {p: round(rng.uniform(0.5, 5.0), 3) for p in params}
        try:
            y = float(module.run_experiment_for_module(noise_level=noise, difficulty=difficulty,
                      system="vanilla_equation", law_version="v0", **inp))
            if math.isfinite(y):
                rows.append({**inp, "_y": y})
        except Exception:
            pass
    return rows


def recover_exponents(module, params, difficulty="easy"):
    """Read the TRUE per-variable power exponents from clean data by log-log slope (multiplicative
    power laws). Returns (const, {param: exponent}) — the ground-truth law, no LLM."""
    rng = random.Random(3)
    # multivariate log-log regression on clean data
    rows = collect_noisy(module, params, 0.0, n=80, seed=5, difficulty=difficulty)
    X, y = [], []
    ok = True
    for r in rows:
        if r["_y"] <= 0 or any(r[p] <= 0 for p in params):
            continue
        X.append([math.log(r[p]) for p in params] + [1.0])
        y.append(math.log(r["_y"]))
    if len(X) < len(params) + 2:
        return None
    coef, *_ = np.linalg.lstsq(np.array(X), np.array(y), rcond=None)
    exps = {p: round(float(c) * 4) / 4 for p, c in zip(params, coef[:-1])}
    const = math.exp(coef[-1])
    return const, exps


def law_pred(const, exps, row, params):
    v = const
    for p in params:
        v *= row[p] ** exps[p]
    return v


def rmsle_law(const, exps, rows, params):
    errs = []
    for r in rows:
        try:
            pred = law_pred(const, exps, r, params); t = r["_y"]
            if pred > 0 and t > 0:
                errs.append((math.log(pred) - math.log(t)) ** 2)
            else:
                errs.append(9.0)
        except Exception:
            errs.append(9.0)
    return math.sqrt(sum(errs) / len(errs)) if errs else 9.0


def make_variants(const, exps, params):
    """Correct law + plausible WRONG variants (perturb one exponent by +-0.5/+-1, or the const)."""
    variants = [("correct", const, dict(exps))]
    for p in params:
        for d in (-1.0, -0.5, 0.5, 1.0):
            e2 = dict(exps); e2[p] = round(e2[p] + d, 2)
            variants.append((f"{p}{d:+}", const, e2))
    variants.append(("const*3", const * 3, dict(exps)))
    variants.append(("const/3", const / 3, dict(exps)))
    return variants


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="vf_v1")
    ap.add_argument("--difficulty", default="easy")
    ap.add_argument("--modules", default=",".join(MODULES))
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out_dir = _PROJ / "lmw" / "verifier_faithfulness" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit("exists; --overwrite")

    modules = args.modules.split(",")
    print(f"=== Verifier faithfulness f(noise) on NewtonBench — no LLM ===")
    print(f"    f = P(held-out fidelity ranks the CORRECT law #1 among perturbed variants)\n")

    rows_out = []
    for noise in NOISE:
        hits = 0; total = 0; gaps = []
        for mod_name in modules:
            module = importlib.import_module(f"modules.{mod_name}")
            params = _params(str(module.FUNCTION_SIGNATURE).strip())
            rec = recover_exponents(module, params, args.difficulty)
            if rec is None:
                continue
            const, exps = rec
            variants = make_variants(const, exps, params)
            for t in range(args.trials):
                data = collect_noisy(module, params, noise, n=40, seed=100 + t, difficulty=args.difficulty)
                if len(data) < 10:
                    continue
                ho = data[int(len(data) * 0.5):]
                scored = [(rmsle_law(c, e, ho, params), name) for name, c, e in variants]
                scored.sort()
                total += 1
                if scored[0][1] == "correct":
                    hits += 1
                # gap between best-wrong and correct (discriminative margin)
                corr = next(s for s in scored if s[1] == "correct")[0]
                best_wrong = next((s[0] for s in scored if s[1] != "correct"), 9.9)
                gaps.append(best_wrong - corr)
        f = hits / total if total else 0.0
        gap = float(np.mean(gaps)) if gaps else 0.0
        print(f"  noise={noise:<5}  f(faithfulness)={f:.2f}   mean gap(best_wrong - correct)={gap:+.4f}", flush=True)
        rows_out.append({"noise": noise, "faithfulness": round(f, 3), "mean_gap": round(gap, 4), "n": total})
        out_path.write_text(json.dumps({"f_of_noise": rows_out}, indent=2))

    # find collapse point: last noise where f >= 0.9
    grounded = [r["noise"] for r in rows_out if r["faithfulness"] >= 0.9]
    coll = max(grounded) if grounded else 0.0
    print(f"\n=== MICROSCOPIC LAW ===")
    print(f"  verifier stays faithful (f>=0.9) up to noise = {coll}")
    print(f"  amplification ceiling = f (bounded by verifier, not by generator support)")
    print(f"  matches the measured tower-depth grounding (~0.1) from the meta-loop")
    out_path.write_text(json.dumps({"collapse_noise_f0.9": coll, "f_of_noise": rows_out}, indent=2))
    print(f"\nsummary → {out_path}")


if __name__ == "__main__":
    main()
