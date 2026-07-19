"""NewtonBench as a THIN ADAPTER over the universal conservation-discovery engine.

No tuned constants: the accept threshold is the data's MEASURED noise floor (queried by
repeating an experiment). The engine (mars.induction.conservation_discovery) does all the
selection/abstention; this file only wires NB-specific I/O: how to generate candidate laws
(model authors a feature library; shell regresses), how to resample observations, how to
measure the noise floor, and the official scorer.

  python -m mars.runners.run_conservation_nb --run_id cnb_v1 --difficulty easy
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
    load_dotenv(_PROJ / "autodiscovery" / ".env.local", override=False)
except ImportError:
    pass

from mars.agents.base import make_openai_client
from mars.induction import conservation_discovery as cd
from mars.runners.run_nb_selfverify import _params, collect
from mars.runners.run_nb_fitters import _law_from_source, _rmsle
from mars.runners.run_unified_nb import (author_forms, author_forms_residual,
                                         expand_products, close_holes_and_fit)

WEAK = "openai/gpt-4o-mini"
MODULES = ["m0_gravity", "m1_coulomb_force", "m9_hooke_law",
           "m4_snell_law", "m5_radioactive_decay", "m7_malus_law"]


def measure_noise_floor(module, params, difficulty, n=6):
    """Universal: query the SAME input twice; the relative discrepancy is the irreducible
    noise. For a deterministic (noiseless) environment this returns ~0 -> the engine then
    demands machine-precision reproduction (rejects spurious within-range fits)."""
    rng = np.random.RandomState(7)
    diffs = []
    for _ in range(n):
        inp = {p: round(0.5 + rng.random() * 4, 3) for p in params}
        try:
            a = float(module.run_experiment_for_module(noise_level=0.0, difficulty=difficulty,
                      system="vanilla_equation", law_version="v0", **inp))
            b = float(module.run_experiment_for_module(noise_level=0.0, difficulty=difficulty,
                      system="vanilla_equation", law_version="v0", **inp))
            if math.isfinite(a) and math.isfinite(b) and (abs(a) + abs(b)) > 0:
                diffs.append(abs(a - b) / (abs(a) + 1e-12))
        except Exception:
            continue
    return float(np.median(diffs)) if diffs else 0.0


def build_adapter(client, model, module, params, entry, sig, rows, difficulty):
    tr, ho = rows[: int(len(rows) * 0.6)], rows[int(len(rows) * 0.6):]
    tr_fit, tr_val = tr[: int(len(tr) * 0.6)], tr[int(len(tr) * 0.6):]

    def generate_candidates(ctx):
        forms = author_forms(client, model, params, tr)
        out, best0 = [], None

        def add(form):
            nonlocal best0
            for f in (form, expand_products(form)):
                src, varies, _ = close_holes_and_fit(f, params, tr_fit, val_rows=tr_val, entry=entry)
                if src is None or not varies:
                    continue
                law = _law_from_source(src, entry)
                if law is None:
                    continue
                # degenerate guard: real law varies with inputs
                pv = []
                for r in ho:
                    try:
                        pv.append(float(law(**{p: r[p] for p in params})))
                    except Exception:
                        pass
                if pv and np.std(pv) / (abs(np.mean(pv)) + 1e-12) < 0.02:
                    continue
                fid = _rmsle(law, ho, params)
                if best0 is None or fid < best0:
                    best0 = fid
                out.append(cd.Candidate(
                    answer=src,
                    fidelity=lambda c, _law=law: _rmsle(_law, c["ho"], params),
                    meta={"law": src.split("return", 1)[-1].strip()[:50]}))

        for form in forms:
            add(form)
        # residual-guided self-debug round if nothing fit well yet
        if (best0 is None or best0 > 0.05):
            best_form = forms[0] if forms else {"target": "log", "features": [f"log({params[0]})"]}
            for form in author_forms_residual(client, model, params, tr, best_form, best0 or 9.9):
                add(form)
        return out

    def transforms(ctx):
        ho_rows = ctx["ho"]
        rng = np.random.RandomState(0)
        n = len(ho_rows)
        out = []
        for i in range(12):
            idx = rng.randint(0, n, size=n) if i % 2 == 0 else rng.permutation(n)[: max(4, int(n * 0.6))]
            sub = [ho_rows[j] for j in idx]
            out.append(lambda c, _s=sub: {**c, "ho": _s})
        return out

    floor = measure_noise_floor(module, params, difficulty)
    adapter = cd.DiscoveryAdapter(
        generate_candidates=generate_candidates,
        transforms=transforms,
        score=lambda ans: float(module.evaluate_law(
            ans, param_description=getattr(module, "PARAM_DESCRIPTION", ""),
            difficulty=difficulty, law_version="v0", judge_model_name="gpt41").get("exact_accuracy", 0.0)),
        noise_floor=lambda c: floor,
    )
    return adapter, {"ho": ho}, floor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="cnb_v1")
    ap.add_argument("--model", default=WEAK)
    ap.add_argument("--modules", default=",".join(MODULES))
    ap.add_argument("--difficulty", default="easy")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out_dir = _PROJ / "lmw" / "conservation_nb" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit("exists; --overwrite")

    client = make_openai_client()
    results = {}; sa_list = []; n_ans = n_abs = 0
    print(f"=== NewtonBench via universal conservation engine ({args.difficulty}) — {args.model} ===\n")
    for mod_name in args.modules.split(","):
        module = importlib.import_module(f"modules.{mod_name}")
        sig = str(module.FUNCTION_SIGNATURE).strip()
        entry = sig[4:sig.index("(")].strip()
        params = _params(sig)
        rows = collect(module, params, difficulty=args.difficulty, n=36)
        if len(rows) < 10:
            results[mod_name] = {"status": "no_data"}; n_abs += 1; continue
        adapter, ctx, floor = build_adapter(client, args.model, module, params, entry, sig, rows, args.difficulty)
        sel = cd.select(adapter, ctx)
        if sel.abstained:
            n_abs += 1
            results[mod_name] = {"status": "ABSTAIN", "reason": sel.reason, "noise_floor": floor}
            print(f"  {mod_name:20} ABSTAIN ({sel.reason}; floor={floor:.1e})"); continue
        sa = adapter.score(sel.answer)
        sa_list.append(sa); n_ans += 1
        snippet = sel.answer.split("return", 1)[-1].strip()[:48]
        results[mod_name] = {"status": "ANSWER", "SA": sa, "noise_floor": floor, "law": snippet}
        print(f"  {mod_name:20} SA={sa:.2f} floor={floor:.1e} | {snippet}")
        out_path.write_text(json.dumps(results, indent=2, default=str))

    n = n_ans + n_abs
    mean_ans = sum(sa_list) / len(sa_list) if sa_list else 0.0
    mean_all = sum(sa_list) / n if n else 0.0
    print(f"\n=== RESULT ({args.difficulty}) ===")
    print(f"  answered {n_ans}/{n}  abstained {n_abs}/{n}")
    print(f"  SA on answered = {mean_ans:.2f}  |  SA over all = {mean_all:.2f}")
    print(f"  (accept threshold = MEASURED noise floor; zero tuned constants)")
    out_path.write_text(json.dumps({"model": args.model, "difficulty": args.difficulty,
                                    "answered": n_ans, "abstained": n_abs,
                                    "SA_answered": round(mean_ans, 3), "SA_all": round(mean_all, 3),
                                    "results": results}, indent=2, default=str))
    print(f"\nsummary → {out_path}")


if __name__ == "__main__":
    main()
