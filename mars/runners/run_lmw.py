"""
Run MARS on LMW (cheap dev-loop, NOT a publishable result).

Used to validate that the multi-agent dialogue actually fires with real
LLMs before moving to paid bench runs. Output goes to lmw/mars_lmw_*.json.

ENV:
  OLS_SEEDS              comma-sep seeds (default 1)
  MARS_GENERATOR_MODEL   inner-LLM (default openai/gpt-4o-mini)
  MARS_REFLECTOR_MODEL   reflector-LLM (default openai/gpt-4o-mini)
  MARS_ABLATIONS         comma-sep tags from {MARS-full, MARS-no-ref,
                         MARS-no-mem, MARS-no-fut, MARS-all-off}
  OLS_MAX_STAGES         per-curriculum stages (default 1 for cheap dev)
"""

from __future__ import annotations

import json
import os
import statistics as st_stats
import sys
import time
from pathlib import Path

_PROJ = Path(__file__).resolve().parent.parent.parent
if str(_PROJ / "lmw") not in sys.path:
    sys.path.insert(0, str(_PROJ / "lmw"))
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

from openworld import build_stage, TIGHTNESS, MAX_STAGES                    # type: ignore
from oracle import Oracle                                                   # type: ignore
from scm import SCM                                                          # type: ignore
from world import World                                                      # type: ignore

from ols.adapters.lmw_adapter import LMWAdapter

from mars.agents.generator import Generator
from mars.agents.memory_selector import MemorySelector
from mars.agents.reflector import Reflector
from mars.coordinator import Coordinator


ABLATIONS = [
    ("MARS-full",
     dict(use_reflector=True,  use_memory_selector=True,  use_futility_detector=True)),
    ("MARS-no-ref",
     dict(use_reflector=False, use_memory_selector=True,  use_futility_detector=True)),
    ("MARS-no-mem",
     dict(use_reflector=True,  use_memory_selector=False, use_futility_detector=True)),
    ("MARS-no-fut",
     dict(use_reflector=True,  use_memory_selector=True,  use_futility_detector=False)),
    ("MARS-all-off",
     dict(use_reflector=False, use_memory_selector=False, use_futility_detector=False)),
]


def run_seed(seed: int, ablation: dict, gen_model: str, ref_model: str,
             max_stages: int = 1) -> dict:
    total_rps = 0.0
    per_stage = []
    reached = 0
    for k in range(max_stages):
        st = build_stage(seed, k)
        budget = round(Oracle(SCM(st, 0)).reference_budget() * TIGHTNESS, 1)
        world = World(structure=st, noise_seed=0, budget=budget)
        adapter = LMWAdapter(
            world,
            regime_shift_at=(st.schema.t_star if st.schema.regime_shift else None),
        )
        gen = Generator(model=gen_model)
        ref = Reflector(model=ref_model)
        sel = MemorySelector(k=5)
        coord = Coordinator(
            adapter=adapter,
            generator=gen,
            reflector=ref,
            memory_selector=sel,
            seed=seed,
            verbose=False,
            **ablation,
        )
        rep = coord.run_episode()
        per_stage.append({
            "k": k,
            "RPS": round(rep.primary, 3),
            "n_turns": rep.n_turns,
            "n_actions": rep.n_actions,
            "n_claims_active": rep.n_claims_active,
            "n_claims_retracted": rep.n_claims_retracted,
            "n_subgoals_done": rep.n_subgoals_done,
            "n_subgoals_abandoned": rep.n_subgoals_abandoned,
            "n_generator_calls": rep.n_generator_calls,
            "n_reflector_calls": rep.n_reflector_calls,
            "verdict_counts": rep.verdict_counts,
            "wall_time_s": round(rep.wall_time_s, 1),
        })
        total_rps += rep.primary
        if rep.primary >= 0.15:
            reached = k + 1
        else:
            break
    return {
        "master_seed": seed,
        "ablation": ablation,
        "reached": reached,
        "total_rps": round(total_rps, 3),
        "per_stage": per_stage,
    }


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    seeds = [int(x) for x in os.environ.get("OLS_SEEDS", "1").split(",") if x.strip()]
    gen_model = os.environ.get("MARS_GENERATOR_MODEL", "openai/gpt-4o-mini")
    ref_model = os.environ.get("MARS_REFLECTOR_MODEL", "openai/gpt-4o-mini")
    ablations_env = os.environ.get("MARS_ABLATIONS", "MARS-full")
    chosen = [a for a in ABLATIONS if a[0] in {x.strip() for x in ablations_env.split(",")}]
    max_stages = int(os.environ.get("OLS_MAX_STAGES", "1"))

    print(f"\n=== MARS × LMW dev smoke ===")
    print(f"  gen={gen_model}  ref={ref_model}  seeds={seeds}  stages={max_stages}")
    print(f"  ablations: {[a[0] for a in chosen]}\n")

    dump = {"generator_model": gen_model, "reflector_model": ref_model,
            "seeds": seeds, "max_stages": max_stages,
            "ablations": [a[0] for a in chosen]}
    for tag, abl in chosen:
        rows = []
        t0 = time.time()
        for s in seeds:
            print(f"[{tag}] seed={s} starting...", flush=True)
            r = run_seed(s, abl, gen_model, ref_model, max_stages=max_stages)
            rows.append(r)
            ps = r["per_stage"]
            print(f"  -> RPS={r['total_rps']:+.3f}, "
                  f"turns={[p['n_turns'] for p in ps]}, "
                  f"claims={[p['n_claims_active'] for p in ps]}, "
                  f"retracts={[p['n_claims_retracted'] for p in ps]}, "
                  f"verdicts={ps[0]['verdict_counts'] if ps else {}}",
                  flush=True)
        dump[tag] = rows
        rps_dist = [r["total_rps"] for r in rows]
        print(f"  {tag:<14} rps.mean={st_stats.fmean(rps_dist):+.3f}  "
              f"(elapsed {time.time()-t0:.0f}s)\n")

    out = _PROJ / "lmw" / f"mars_lmw_{gen_model.replace('/','-')}_{len(seeds)}s.json"
    with open(out, "w") as f:
        json.dump(dump, f, indent=1)
    print(f"[written {out}]")


if __name__ == "__main__":
    main()
