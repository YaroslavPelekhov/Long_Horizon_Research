"""META-TOWER curriculum: measure the DEPTH of grounded recursive verification on NewtonBench.

The tower's L1 selects a METHOD by a gold-free signal (held-out predictive fidelity). Theory:
at noise ν the noise floor is ~ν, so the gold-free signal can tell the correct method from a
wrong-but-close one only while the fidelity GAP exceeds ν. As ν rises the gap collapses and the
meta-selection DECOUPLES from truth. We sweep ν (the "ever more complex metric" axis) and find
the level where gold-free selection stops matching the gold-best method — that breaking point IS
the measured verifiable structure / usable tower depth of the environment.

  python -m mars.runners.run_meta_curriculum_nb --run_id mc_v1 --model openai/gpt-4o-mini
"""
from __future__ import annotations

import argparse
import importlib
import json
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
    load_dotenv(_PROJ / "autodiscovery" / ".env.local", override=False)
except ImportError:
    pass

from mars.agents.base import make_openai_client
from mars.runners.run_nb_selfverify import _params
from mars.runners.run_meta_tower_nb import METHODS

WEAK = "openai/gpt-4o-mini"
MODULES = ["m0_gravity", "m9_hooke_law", "m1_coulomb_force"]
NOISE_LEVELS = [0.0, 0.02, 0.05, 0.1, 0.2, 0.4]


def collect_noisy(module, params, noise, difficulty="easy", n=36, seed=11):
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        inp = {p: round(rng.uniform(0.5, 5.0), 3) for p in params}
        try:
            y = float(module.run_experiment_for_module(noise_level=noise, difficulty=difficulty,
                      system="vanilla_equation", law_version="v0", **inp))
            if y == y:
                rows.append({**inp, "_y": y})
        except Exception:
            continue
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", default="mc_v1")
    ap.add_argument("--model", default=WEAK)
    ap.add_argument("--difficulty", default="easy")
    ap.add_argument("--modules", default=",".join(MODULES))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    out_dir = _PROJ / "lmw" / "meta_curriculum_nb" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    if out_path.exists() and not args.overwrite:
        raise SystemExit("exists; --overwrite")

    client = make_openai_client()
    modules = args.modules.split(",")
    print(f"=== META-TOWER curriculum on NewtonBench — {args.model} ===")
    print(f"    sweep noise; find where gold-free method-selection decouples from gold-best\n")

    rows_by_level = []
    grounded_until = None
    for noise in NOISE_LEVELS:
        rmsle_m = {m: [] for m in METHODS}
        sa_m = {m: [] for m in METHODS}
        for mod_name in modules:
            module = importlib.import_module(f"modules.{mod_name}")
            sig = str(module.FUNCTION_SIGNATURE).strip()
            entry = sig[4:sig.index("(")].strip()
            params = _params(sig)
            rows = collect_noisy(module, params, noise, args.difficulty)
            if len(rows) < 12:
                continue
            for mname, fn in METHODS.items():
                try:
                    rm, sa, _src = fn(client, args.model, module, params, entry, sig, rows, args.difficulty)
                except Exception:
                    rm, sa = 9.9, 0.0
                rmsle_m[mname].append(rm)
                sa_m[mname].append(sa)
        mean_rmsle = {m: float(np.mean(v)) if v else 9.9 for m, v in rmsle_m.items()}
        mean_sa = {m: float(np.mean(v)) if v else 0.0 for m, v in sa_m.items()}
        pick = min(mean_rmsle, key=mean_rmsle.get)            # gold-free
        gold = max(mean_sa, key=mean_sa.get)                  # validation
        match = (pick == gold) and (mean_sa[gold] > 0)        # grounded == matches AND something solved
        # fidelity gap between best and 2nd-best method (the discriminative margin)
        srt = sorted(mean_rmsle.values())
        gap = srt[1] - srt[0] if len(srt) > 1 else 0.0
        if match:
            grounded_until = noise
        print(f"  noise={noise:<4}  pick='{pick:11}' gold='{gold:11}' "
              f"{'GROUNDED ✓' if match else 'DECOUPLED ✗'}  gap={gap:.3f}  "
              f"best_SA={mean_sa[gold]:.2f}", flush=True)
        rows_by_level.append({"noise": noise, "pick": pick, "gold": gold, "grounded": match,
                              "gap": round(gap, 4), "mean_rmsle": {k: round(v, 4) for k, v in mean_rmsle.items()},
                              "mean_sa": {k: round(v, 3) for k, v in mean_sa.items()}})
        out_path.write_text(json.dumps({"levels": rows_by_level}, indent=2, default=str))

    print(f"\n=== TOWER DEPTH MEASUREMENT ===")
    print(f"  gold-free method-selection stays grounded up to noise = {grounded_until}")
    print(f"  beyond that the meta-signal decouples from truth (structure exhausted)")
    n_grounded = sum(1 for r in rows_by_level if r["grounded"])
    print(f"  usable curriculum rungs: {n_grounded}/{len(NOISE_LEVELS)}")
    out_path.write_text(json.dumps({"grounded_until_noise": grounded_until,
                                    "usable_rungs": n_grounded, "levels": rows_by_level},
                                   indent=2, default=str))
    print(f"\nsummary → {out_path}")


if __name__ == "__main__":
    main()
