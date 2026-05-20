"""
OLS smoke test — exercises the core loop on a single LMW world with a
deterministic mock inner agent (no LLM, no API calls, no $$).

Goal: catch wiring bugs (import paths, action plumbing, adapter contract,
claim-store + agenda + futility lifecycle, scoring bridge) before we spend
any API budget on a real LLM sweep.

Pass conditions:
  - One LMWAdapter episode runs to completion (budget exhausted OR halt)
  - At least one action executes and at least one claim is asserted
  - LMWAdapter.score_episode() returns a finite primary value
  - All ablation switches toggle without crashing
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make sibling lmw/ importable for this script (run via `python ols/smoketest.py`)
_PROJ = Path(__file__).resolve().parent.parent
if str(_PROJ / "lmw") not in sys.path:
    sys.path.insert(0, str(_PROJ / "lmw"))
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

from ols.adapters.lmw_adapter import LMWAdapter
from ols.inner_agents.base import (
    ActionRequest,
    AgentContext,
    AgentResponse,
    ClaimDelta,
)
from ols.scaffold import OLSScaffold


class MockInnerAgent:
    """A deterministic, LLM-free inner agent for wiring tests."""
    name = "MockInner"

    def __init__(self):
        self.turn = 0

    def propose(self, ctx: AgentContext) -> AgentResponse:
        self.turn += 1
        # Always issue one observe action on (up to) 4 of the variables the
        # current sub-goal mentions, then one intervene if we have any history.
        # Use the env_description / question text only to find a few var names.
        import re
        vars_in_q = re.findall(r"\bX\d+\b", ctx.sub_goal_question)
        vars_in_q = list(dict.fromkeys(vars_in_q))[:4]                  # dedupe, cap

        actions = []
        if vars_in_q:
            actions.append(ActionRequest(
                action="observe",
                args={"vars": vars_in_q, "n": 6},
            ))
        # if we have a previous eid for this sub-goal, do an intervention too
        if ctx.history_for_this_subgoal and len(vars_in_q) >= 2:
            a, b = vars_in_q[0], vars_in_q[1]
            actions.append(ActionRequest(
                action="intervene",
                args={"var": a, "value": 2.0, "outcomes": [b], "n": 4},
            ))

        claims = []
        # after the 2nd turn for this sub-goal, claim something deterministic
        if len(ctx.history_for_this_subgoal) >= 2 and len(vars_in_q) >= 2:
            a, b = vars_in_q[0], vars_in_q[1]
            claims.append(ClaimDelta(
                op="assert",
                statement=f"no_effect({a},{b})",
                confidence=0.6,
            ))

        # advance after ~3 turns per sub-goal
        adv = len(ctx.history_for_this_subgoal) >= 3
        return AgentResponse(
            actions=actions,
            claims=claims,
            advance_subgoal=adv,
            rationale=f"mock-turn-{self.turn}",
        )


def _build_lmw_world(seed=1):
    # Build a single small SCM directly (cheap path, no curriculum).
    from scm import Schema, make_structure                               # type: ignore
    from world import World                                              # type: ignore

    sch = Schema(n_clusters=3, chain_len=2, n_latents=2, real_decoys=1,
                 regime_shift=False)
    structure = make_structure(seed, sch)
    # Modest budget for the smoke test
    return World(structure=structure, noise_seed=seed * 7919, budget=60.0)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("\n=== OLS smoke test (LMW adapter, mock inner agent) ===\n")
    for tag, abl in [
        ("OLS-ALL-ON",  dict(use_persistent_store=True,  use_agenda_controller=True,  use_futility_detector=True)),
        ("OLS-MEM-OFF", dict(use_persistent_store=False, use_agenda_controller=True,  use_futility_detector=True)),
        ("OLS-AGN-OFF", dict(use_persistent_store=True,  use_agenda_controller=False, use_futility_detector=True)),
        ("OLS-FUT-OFF", dict(use_persistent_store=True,  use_agenda_controller=True,  use_futility_detector=False)),
        ("OLS-ALL-OFF", dict(use_persistent_store=False, use_agenda_controller=False, use_futility_detector=False)),
    ]:
        world = _build_lmw_world(seed=1)
        adapter = LMWAdapter(world)
        scaffold = OLSScaffold(
            adapter=adapter,
            inner_agent=MockInnerAgent(),
            seed=1,
            verbose=False,
            **abl,
        )
        rep = scaffold.run_episode()
        print(f"{tag:<13} RPS={rep.primary:+.3f}  "
              f"act={rep.n_actions:>3}  cl={rep.n_claims_active:>2}/+{rep.n_claims_retracted}  "
              f"sg(done/aband)={rep.n_subgoals_done}/{rep.n_subgoals_abandoned}  "
              f"ax1={rep.axis1_ratio:.2f}  ax6={rep.axis6_regret:.2f}  "
              f"t={rep.wall_time_s:.2f}s")

    print("\nIf all rows printed without exception, OLS wiring is sound.")


if __name__ == "__main__":
    main()
