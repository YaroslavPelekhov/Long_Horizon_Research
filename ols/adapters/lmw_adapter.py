"""
LMW adapter — wraps lmw.world.World as a ResearchEnvAdapter.

This is the home-bench plug. Same World API the existing agents see (observe,
intervene, budget, log); OLS just gets a uniform typed surface.

Scoring uses the existing lmw.scorer.score() function with the LMW Oracle
(the SCM-known ground-truth), so OLS reports the EXACT same RPS that
run_open_llm.py / run_autodisc.py report — no re-derivation, no fork.
"""

from __future__ import annotations

import sys
from pathlib import Path

# import the lmw/ package as siblings (no install required)
_HERE = Path(__file__).resolve().parent
_LMW = _HERE.parent.parent / "lmw"
if str(_LMW) not in sys.path:
    sys.path.insert(0, str(_LMW))

from world import BudgetExhausted as _LmwBudgetExhausted, World          # type: ignore
from oracle import Oracle                                                  # type: ignore
from scm import SCM, make_structure                                       # type: ignore
from scorer import score as lmw_score                                      # type: ignore

from ols.adapters.base import (
    BudgetExhausted,
    EnvHandle,
    ResearchEnvAdapter,
)
from ols.core.types import (
    ActionSpec,
    Claim,
    ClaimVerdict,
    ExperimentResult,
)


# Recreate the lmw FACTS-style canonical statements but keep them as opaque
# strings to OLS; the adapter does any parsing.
_LMW_ACTIONS = [
    ActionSpec(
        name="observe",
        arg_schema={"vars": "list[str]", "n": "int"},
        cost_estimate=4.0,
        description=("Passive observation of the listed variables for n samples. "
                     "Cost ≈ 0.5·n. Returns per-sample dicts and pairwise corrs."),
    ),
    ActionSpec(
        name="intervene",
        arg_schema={"var": "str", "value": "float", "outcomes": "list[str]", "n": "int"},
        cost_estimate=8.0,
        description=("Hard intervention do(var=value); record outcome variables "
                     "for n samples. Cost ≈ 1.0·n. Use ± values to estimate a "
                     "causal effect on a target."),
    ),
]


class LMWAdapter(ResearchEnvAdapter):
    """Wraps an LMW World as a ResearchEnvAdapter.

    Construct from an existing World (typical: caller built it via
    lmw.harness or lmw.openworld), OR from (structure_seed, noise_seed,
    schema) for standalone use.
    """

    def __init__(self, world: World, oracle: Oracle | None = None,
                 regime_shift_at: float | None = None,
                 name: str = "LMWAdapter"):
        self.world = world
        self.oracle = oracle if oracle is not None else Oracle(world._scm)
        self.regime_shift_at = regime_shift_at
        self._name = name

    # -- handle ---------------------------------------------------------------

    def handle(self) -> EnvHandle:
        clusters = self.world._scm.structure.subdomains
        subdomains: list[tuple[str, str]] = []
        for cname, vs in sorted(clusters.items()):
            subdomains.append((
                cname,
                f"Identify any causal structure among the variables of "
                f"sub-domain {cname}: {vs}. Report findings as claims of the form "
                f"causal(X,Y,+|-), no_effect(X,Y), confounded(X,Y,Z), or "
                f"mean(X)~const.",
            ))

        desc = (
            "LMW (Latent Mechanism World) — a synthetic causal environment with "
            "several disjoint variable clusters. Some clusters are dud (no real "
            "structure), some have a single simple causal edge, and at least one "
            "has a hidden chain plus a tempting-but-false correlation through a "
            "latent confounder. Tools: observe() for correlations, intervene() "
            f"for ground-truth causal effects. Total budget = "
            f"{self.world.budget_total}."
        )
        return EnvHandle(
            description=desc,
            subdomains=subdomains,
            actions=list(_LMW_ACTIONS),
            budget_total=self.world.budget_total,
        )

    def budget_left(self) -> float:
        return float(self.world.budget_left)

    # -- execute --------------------------------------------------------------

    def execute(self, action: str, args: dict) -> ExperimentResult:
        try:
            if action == "observe":
                vs = list(args.get("vars", []))
                n = int(args.get("n", 8))
                e = self.world.observe(vs, n=n)
                summary = {"corrs": {f"{a}~{b}": e.corr(a, b)
                                     for i, a in enumerate(vs)
                                     for b in vs[i + 1:]
                                     if a in e.samples[0] and b in e.samples[0]},
                           "means": {v: e.mean(v) for v in vs
                                     if e.samples and v in e.samples[0]},
                           "n": len(e.samples)}
                return ExperimentResult(
                    eid=e.eid, action="observe", args=args,
                    cost=0.5 * n, raw=e, summary=summary,
                )
            elif action == "intervene":
                var = str(args["var"])
                value = float(args["value"])
                outcomes = list(args.get("outcomes", []))
                n = int(args.get("n", 6))
                e = self.world.intervene(var, value, outcomes, n=n)
                summary = {"means": {v: e.mean(v) for v in outcomes
                                     if e.samples and v in e.samples[0]},
                           "n": len(e.samples), "var": var, "value": value}
                return ExperimentResult(
                    eid=e.eid, action="intervene", args=args,
                    cost=1.0 * n, raw=e, summary=summary,
                )
            else:
                # Unknown action — burn no budget; signal via summary.
                return ExperimentResult(
                    eid=-1, action=action, args=args, cost=0.0, raw=None,
                    summary={"error": f"unknown action: {action}"},
                )
        except _LmwBudgetExhausted as exc:
            raise BudgetExhausted(str(exc)) from exc

    # -- ground-truth check ---------------------------------------------------

    def verify_claim(self, claim: Claim) -> ClaimVerdict | None:
        try:
            t = self.oracle.is_true(claim.statement, budget=self.world.budget_total)
            return ClaimVerdict(true=bool(t), confidence=1.0)
        except Exception:
            return None

    # -- scoring --------------------------------------------------------------

    def score_episode(self, claim_store_active, final_artifact=None) -> dict:
        """Use the lmw scorer for headline RPS.

        Bridge: OLS active claims → a lmw.ClaimStore-shaped object the scorer
        consumes. We rebuild a small lmw-side store with the same statements
        and provenance, then call scorer.score on the world + oracle.
        """
        from claims import ClaimStore as _LmwStore                       # type: ignore

        bridge = _LmwStore()
        for c in claim_store_active:
            bridge.assert_claim(c.statement, c.confidence,
                                list(c.provenance), c.budget_stamp)
        # mimic axis-1 agenda log w/ a single snapshot (the lmw scorer uses
        # bridge.agenda_log only as a small input — our axis-1 metric is OLS-side)
        bridge.set_agenda([], self.world.budget_total - self.world.budget_left)
        out = lmw_score(self.world, bridge, self.oracle,
                        regime_shift_at=self.regime_shift_at)
        return {
            "primary": out["RPS"],
            "RPS": out["RPS"],
            "axis1_agenda_ratio": out.get("axis1_agenda_ratio", 0.0),
            "axis2_retention": out.get("axis2_retention", 0.0),
            "axis2_revision": out.get("axis2_revision", 0.0),
            "axis6_deadend_regret": out.get("axis6_deadend_regret", 0.0),
            "integrity_penalty": out.get("integrity_penalty", 0.0),
            "final_true": out.get("final_true", 0.0),
        }
