"""
Run OLS overlay on Open-Ended LMW (5 master seeds × 2 conditions × N ablations).

For each (seed, condition, ablation) we:
  1. Build stage-k world via lmw.openworld.build_stage (the SAME function the
     existing leaderboard entries use — RPS is directly comparable).
  2. Wrap it in LMWAdapter.
  3. Drive OLSScaffold + OLSInnerLLM (gpt-4o-mini by default).
  4. If `persistent=True`, carry the OLS ClaimStore across stages.
  5. Advance to stage k+1 iff stage-k RPS ≥ τ (the standard mastery gate).

Outputs JSON to lmw/ols_lmw_<model>_<seeds>s.json so it lives alongside the
other adapter dumps and metrics_multi.py can ingest it later.
"""

from __future__ import annotations

import json
import os
import sys
import statistics as st_stats
import time
from pathlib import Path

_PROJ = Path(__file__).resolve().parent.parent.parent
if str(_PROJ / "lmw") not in sys.path:
    sys.path.insert(0, str(_PROJ / "lmw"))
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

# lmw imports
from openworld import build_stage, consistency_selftest, TIGHTNESS, MAX_STAGES   # type: ignore
from oracle import Oracle                                                         # type: ignore
from scm import SCM                                                               # type: ignore
from world import World                                                           # type: ignore

# ols imports
from ols.adapters.lmw_adapter import LMWAdapter
from ols.core.claim_store import ClaimStore
from ols.inner_agents.llm_inner import OLSInnerLLM
from ols.scaffold import OLSScaffold


def _stage_view(store: ClaimStore, k: int) -> ClaimStore:
    """Return a view containing only claims that mention stage-k variables.

    Mirrors lmw.openworld._stage_view: cross-stage namespaced theories are
    valid and persist, but ONLY this stage's claims are scored against the
    stage-k oracle.
    """
    v = ClaimStore()
    needle = f"s{k}."
    v.claims = [c for c in store.claims if needle in c.statement]
    return v


def run_seed(
    master_seed: int,
    ablation: dict,
    model: str,
    tau: float = 0.15,
    max_stages: int = MAX_STAGES,
    verbose: bool = False,
) -> dict:
    """Run one full curriculum for one master_seed.

    Cross-stage persistence follows ablation['use_persistent_store']: if ON,
    the OLS ClaimStore is carried from stage k → k+1; if OFF, each stage
    starts with a fresh store (the cross-horizon axis-2 ablation).
    """
    persistent = ablation["use_persistent_store"]
    carry = ClaimStore() if persistent else None     # init persistent carry once
    inner = OLSInnerLLM(model=model)
    reached = 0
    total_rps = 0.0
    per_stage = []

    for k in range(max_stages):
        st = build_stage(master_seed, k)
        budget = round(Oracle(SCM(st, 0)).reference_budget() * TIGHTNESS, 1)
        world = World(structure=st, noise_seed=0, budget=budget)
        adapter = LMWAdapter(
            world,
            regime_shift_at=(st.schema.t_star if st.schema.regime_shift else None),
        )

        scaffold = OLSScaffold(
            adapter=adapter,
            inner_agent=inner,
            carry_store=carry,                       # None when not persistent
            seed=master_seed,
            verbose=verbose,
            **ablation,
        )
        rep = scaffold.run_episode()
        # The scaffold's internal store is exposed via .last_store. Re-score
        # with only THIS stage's claims (namespaced sk.*) to match the
        # curriculum semantics of lmw.openworld._stage_view.
        used_store = scaffold.last_store or ClaimStore()
        stage_only = _stage_view(used_store, k)

        from scorer import score as lmw_score                                # type: ignore
        from claims import ClaimStore as _LmwStore                            # type: ignore
        bridge = _LmwStore()
        for c in stage_only.active():
            bridge.assert_claim(c.statement, c.confidence,
                                list(c.provenance), c.budget_stamp)
        bridge.set_agenda([], world.budget_total - world.budget_left)
        oc = Oracle(world._scm)
        rs = st.schema.t_star if st.schema.regime_shift else None
        sd = lmw_score(world, bridge, oc, regime_shift_at=rs)
        stage_rps = sd["RPS"]

        # Carry forward only when persistent
        if persistent:
            carry = used_store

        per_stage.append({
            "k": k,
            "RPS": round(stage_rps, 3),
            "axis1": round(sd.get("axis1_agenda_ratio", 0.0), 3),
            "n_actions": rep.n_actions,
            "n_claims_active": rep.n_claims_active,
            "ols_axis1_ratio": round(rep.axis1_ratio, 3),
            "ols_axis6_regret": round(rep.axis6_regret, 3),
            "n_subgoals_done": rep.n_subgoals_done,
            "n_subgoals_abandoned": rep.n_subgoals_abandoned,
        })
        total_rps += stage_rps
        if stage_rps >= tau:
            reached = k + 1
        else:
            break

    return {
        "master_seed": master_seed,
        "persistent": persistent,
        "ablation": ablation,
        "reached": reached,
        "total_rps": round(total_rps, 3),
        "per_stage": per_stage,
    }


# -- run sweep ---------------------------------------------------------------

# v0.2: OLS-full enables the require_claim_before_advance gate so the
# inner LLM cannot skip past sub-goals without populating the ClaimStore.
# Diagnosis from v0.1 was that gpt-4o-mini bypassed claim assertion when
# the overlay's machinery was optional. The gate makes it load-bearing.
ABLATIONS = [
    ("OLS-full",
     dict(use_persistent_store=True, use_agenda_controller=True,
          use_futility_detector=True, require_claim_before_advance=True,
          min_claims_per_subgoal=1)),
    # Isolates the v0.2 directive-prompt contribution from the gate's
    # structural enforcement. All three OLS modules ON; only the gate is OFF.
    # If OLS-full-no-gate ≈ OLS-all-off → ALL the gain comes from the gate.
    # If OLS-full-no-gate ≈ OLS-full → ALL the gain comes from the modules
    # themselves (prompt + structure suffice without enforcement).
    ("OLS-full-no-gate",
     dict(use_persistent_store=True, use_agenda_controller=True,
          use_futility_detector=True, require_claim_before_advance=False)),
    ("OLS-mem-off",
     dict(use_persistent_store=False, use_agenda_controller=True,
          use_futility_detector=True, require_claim_before_advance=True,
          min_claims_per_subgoal=1)),
    ("OLS-agn-off",
     dict(use_persistent_store=True, use_agenda_controller=False,
          use_futility_detector=True, require_claim_before_advance=True,
          min_claims_per_subgoal=1)),
    ("OLS-fut-off",
     dict(use_persistent_store=True, use_agenda_controller=True,
          use_futility_detector=False, require_claim_before_advance=True,
          min_claims_per_subgoal=1)),
    ("OLS-all-off",
     dict(use_persistent_store=False, use_agenda_controller=False,
          use_futility_detector=False, require_claim_before_advance=False)),
]

# Mapping: persistent flag tells the curriculum loop whether to carry store
# across stages. Inside-stage ablations are controlled by `ablation` dict.
# For OLS-full and OLS-agn-off etc, persistent follows use_persistent_store.


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    seeds_env = os.environ.get("OLS_SEEDS", "1,2,3")
    seeds = [int(x) for x in seeds_env.split(",") if x.strip()]
    model = os.environ.get("OLS_INNER_MODEL", "openai/gpt-4o-mini")
    ablations = os.environ.get("OLS_ABLATIONS", "OLS-full,OLS-all-off")
    chosen = [a for a in ABLATIONS if a[0] in {x.strip() for x in ablations.split(",")}]
    max_stages = int(os.environ.get("OLS_MAX_STAGES", str(MAX_STAGES)))

    consistency_selftest()
    print(f"\n=== OLS × Open-Ended LMW (model={model}, seeds={seeds}) ===\n")

    dump = {"model": model, "seeds": seeds, "ablations": [a[0] for a in chosen]}
    for tag, abl in chosen:
        rows = []
        t0 = time.time()
        for s in seeds:
            print(f"[{tag}] seed={s} starting...", flush=True)
            r = run_seed(s, abl, model, max_stages=max_stages)
            rows.append(r)
            print(f"  -> reached={r['reached']}, total_rps={r['total_rps']:+.3f}, "
                  f"per_stage={[(p['k'], p['RPS']) for p in r['per_stage']]}",
                  flush=True)
        depth_dist = [r["reached"] for r in rows]
        rps_dist = [r["total_rps"] for r in rows]
        dump[tag] = rows
        print(f"  {tag:<14} depth.mean={st_stats.fmean(depth_dist):.2f}  "
              f"rps.mean={st_stats.fmean(rps_dist):+.3f}  "
              f"depths={depth_dist}  rps={rps_dist}  "
              f"(elapsed {time.time()-t0:.0f}s)\n")

    suffix = f"{model.replace('/','-').replace(':','_')}_{len(seeds)}s"
    out_path = _PROJ / "lmw" / f"ols_lmw_{suffix}.json"
    with open(out_path, "w") as f:
        json.dump(dump, f, indent=1)
    print(f"[written {out_path}]")


if __name__ == "__main__":
    main()
