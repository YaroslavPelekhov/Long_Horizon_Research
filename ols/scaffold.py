"""
OLSScaffold — the overlay control loop.

Drives any InnerAgent against any ResearchEnvAdapter while owning the three
outer-loop capabilities (ClaimStore, AgendaController, FutilityDetector).

The episode loop:

    1. seed agenda from adapter.handle().subdomains
    2. while budget_left > 0 and agenda has active sub-goals:
        a. pick next sub-goal via AgendaController.pick_next()
        b. inner_agent.propose(ctx) → response
        c. for each action: adapter.execute(...) → eid + result
        d. for each claim delta: ClaimStore.assert / retract; AgendaController.record_claim
        e. FutilityDetector.check() → abandon stragglers
        f. if response.advance_subgoal: agenda.mark_done(g)
        g. if response.halt: break
    3. (optional) re-test pass for due claims (axis-2 revision under regime shift)
    4. adapter.score_episode(active_claims, final_artifact) → primary metric

Ablations exposed for the same-seed counterfactual harness in
lmw/scorer.py spirit:
    - use_persistent_store : axis-2 ablation (memory)
    - use_agenda_controller : axis-1 ablation (agenda)
    - use_futility_detector : axis-6 ablation (abandon)
All three OFF ≈ a thin loop over an inner LLM, no outer-loop scaffold.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ols.adapters.base import BudgetExhausted, ResearchEnvAdapter
from ols.core.abandon import FutilityDetector
from ols.core.agenda import AgendaController, SubGoalStatus
from ols.core.claim_store import ClaimStore
from ols.core.types import AgendaItem, Claim
from ols.inner_agents.base import (
    ActionRequest,
    AgentContext,
    AgentResponse,
    ClaimDelta,
    InnerAgent,
)


@dataclass
class EpisodeReport:
    """The artifact OLS hands the runner / paper / sweep harness."""
    primary: float                       # adapter-defined headline (RPS / HMS / etc)
    score_dict: dict                     # full adapter score breakdown
    n_actions: int
    n_claims_active: int
    n_claims_retracted: int
    n_subgoals_done: int
    n_subgoals_abandoned: int
    axis1_ratio: float                   # OLS-side agenda-productivity
    axis6_regret: float                  # OLS-side dead-end regret
    wall_time_s: float
    ablations: dict[str, bool]
    inner_agent_name: str
    adapter_name: str
    seed: int | None = None
    history: list[dict] = field(default_factory=list)
    # per-sub-goal log: {"sub_goal","actions","claims_added","budget_spent"}


@dataclass
class OLSScaffold:
    adapter: ResearchEnvAdapter
    inner_agent: InnerAgent
    # ablation switches (axes 1, 2, 6 from the field map)
    use_persistent_store: bool = True
    use_agenda_controller: bool = True
    use_futility_detector: bool = True
    # hyperparameters
    theta_futile: float = 0.30
    max_inner_calls_per_subgoal: int = 6
    max_total_inner_calls: int = 48
    do_final_retest: bool = True
    # for cross-episode reuse (Open-Ended LMW style continuity)
    carry_store: ClaimStore | None = None
    seed: int | None = None
    verbose: bool = False
    # populated after run_episode() so the caller can inspect / carry forward
    last_store: ClaimStore | None = None

    def run_episode(self) -> EpisodeReport:
        t0 = time.time()
        adapter = self.adapter

        # 1. ClaimStore + Agenda + Futility init
        store = (self.carry_store if (self.use_persistent_store and self.carry_store)
                 else ClaimStore())
        self.last_store = store
        agenda = AgendaController()
        futility = FutilityDetector(theta_futile=self.theta_futile)

        handle = adapter.handle()
        # Seed agenda — even with controller OFF, the inner agent needs SOME
        # sub-goal partition; with controller OFF we simply iterate in declaration
        # order and never re-prioritize.
        for sg, q in handle.subdomains:
            agenda.add(AgendaItem(sub_goal=sg, question=q, rationale=(
                "controlled" if self.use_agenda_controller else "imposed"
            )), adapter.budget_spent())

        history: list[dict] = []
        per_sg_history: dict[str, list[dict]] = {sg: [] for sg, _ in handle.subdomains}
        total_inner_calls = 0
        eid_to_subgoal: dict[int, str] = {}
        eid_to_claim_idx: dict[int, list[int]] = {}

        # 2. Main loop
        try:
            iter_order = [sg for sg, _ in handle.subdomains]
            iter_idx = 0

            while adapter.budget_left() > 0 and total_inner_calls < self.max_total_inner_calls:

                # 2a. Pick next sub-goal
                if self.use_agenda_controller:
                    current_sg = agenda.pick_next()
                else:
                    # ablation: cycle through declaration-order, skipping done/abandoned
                    found = None
                    for _ in range(len(iter_order)):
                        sg = iter_order[iter_idx % len(iter_order)]
                        iter_idx += 1
                        st = agenda.status.get(sg)
                        if st not in (SubGoalStatus.DONE, SubGoalStatus.ABANDONED):
                            found = sg
                            break
                    current_sg = found

                if current_sg is None:
                    break                                # agenda fully resolved

                agenda.promote(current_sg, SubGoalStatus.PURSUE, adapter.budget_spent())
                question = agenda.items[current_sg].question

                # 2b. Build context and call inner agent
                ctx = AgentContext(
                    sub_goal=current_sg,
                    sub_goal_question=question,
                    env_description=handle.description,
                    available_actions=handle.actions,
                    theory_state_view=(
                        store.render_for_agent() if self.use_persistent_store
                        else "(theory state disabled — −mem ablation)"
                    ),
                    agenda_view=(
                        agenda.render_for_agent() if self.use_agenda_controller
                        else "(agenda controller disabled)"
                    ),
                    budget_remaining=adapter.budget_left(),
                    budget_total=handle.budget_total,
                    history_for_this_subgoal=list(per_sg_history[current_sg][-6:]),
                )

                response: AgentResponse = self.inner_agent.propose(ctx)
                total_inner_calls += 1

                if self.verbose:
                    print(f"[OLS] sg={current_sg} t={total_inner_calls} "
                          f"actions={len(response.actions)} "
                          f"claims={len(response.claims)} "
                          f"halt={response.halt} adv={response.advance_subgoal}")

                # 2c. Execute actions
                new_eids: list[int] = []
                for ar in response.actions:
                    try:
                        cost_before = adapter.budget_spent()
                        result = adapter.execute(ar.action, ar.args)
                    except BudgetExhausted:
                        raise
                    new_eids.append(result.eid)
                    eid_to_subgoal[result.eid] = current_sg
                    agenda.record_spend(
                        current_sg, max(0.0, adapter.budget_spent() - cost_before)
                    )
                    per_sg_history[current_sg].append({
                        "action": ar.action,
                        "args": ar.args,
                        "eid": result.eid,
                        "summary": result.summary,
                    })

                # 2d. Process claim deltas
                for cd in response.claims:
                    if cd.op == "retract":
                        store.retract(cd.statement, adapter.budget_spent())
                    else:
                        # default provenance = new eids from THIS turn
                        prov = cd.provenance or new_eids
                        idx = store.assert_claim(
                            statement=cd.statement,
                            confidence=cd.confidence,
                            provenance=prov,
                            budget=adapter.budget_spent(),
                            sub_goal=current_sg,
                            depends_on=cd.depends_on,
                        )
                        eid_to_claim_idx.setdefault(idx, []).extend(prov)
                        agenda.record_claim(current_sg)

                # 2e. Futility check
                if self.use_futility_detector:
                    for fg in futility.check(
                        agenda, handle.budget_total,
                        adapter.budget_left(), adapter.budget_spent(),
                    ):
                        agenda.abandon(fg, adapter.budget_spent())

                # 2f. Advance sub-goal if inner agent says so
                if response.advance_subgoal:
                    agenda.mark_done(current_sg, adapter.budget_spent())

                # 2g. Halt
                if response.halt:
                    break

                # Soft cap on per-sub-goal inner calls (defense-in-depth)
                if len([h for h in per_sg_history[current_sg]
                        if "action" in h]) >= self.max_inner_calls_per_subgoal \
                   and not response.advance_subgoal:
                    agenda.mark_done(current_sg, adapter.budget_spent())

                history.append({
                    "t": total_inner_calls,
                    "sub_goal": current_sg,
                    "n_actions": len(response.actions),
                    "n_claims": len(response.claims),
                    "rationale": response.rationale[:200] if response.rationale else "",
                })

        except BudgetExhausted:
            pass

        # 3. Final re-test pass (axis-2 revision under regime shift)
        if self.use_persistent_store and self.do_final_retest:
            self._final_retest(adapter, store, eid_to_subgoal)

        # 4. Score the episode via the adapter
        active = store.active()
        score = adapter.score_episode(active, final_artifact=None)
        primary = score.get("primary", score.get("RPS", score.get("HMS", 0.0)))

        n_done = sum(1 for g in agenda.items
                     if agenda.status.get(g) == SubGoalStatus.DONE)
        n_aband = sum(1 for g in agenda.items
                      if agenda.status.get(g) == SubGoalStatus.ABANDONED)

        report = EpisodeReport(
            primary=primary,
            score_dict=score,
            n_actions=sum(1 for h in history for _ in range(h["n_actions"])),
            n_claims_active=len(active),
            n_claims_retracted=len([c for c in store.claims if c.status == "retracted"]),
            n_subgoals_done=n_done,
            n_subgoals_abandoned=n_aband,
            axis1_ratio=agenda.axis1_ratio(),
            axis6_regret=futility.regret(agenda, handle.budget_total),
            wall_time_s=time.time() - t0,
            ablations={
                "use_persistent_store": self.use_persistent_store,
                "use_agenda_controller": self.use_agenda_controller,
                "use_futility_detector": self.use_futility_detector,
            },
            inner_agent_name=getattr(self.inner_agent, "name", type(self.inner_agent).__name__),
            adapter_name=type(self.adapter).__name__,
            seed=self.seed,
            history=history,
        )
        return report

    # -- internals ------------------------------------------------------------

    def _final_retest(
        self,
        adapter: ResearchEnvAdapter,
        store: ClaimStore,
        eid_to_sg: dict[int, str],
    ) -> None:
        """Domain-agnostic re-test pass.

        Default policy: if the adapter has a per-claim verifier (verify_claim
        returns non-None), use it to confirm or retract every active claim.
        For adapters without an oracle (DiscoveryBench), this is a no-op —
        re-test happens via the end-of-episode LLM-judge.
        """
        if adapter.budget_left() <= 0:
            return
        for c in list(store.active()):
            verdict = adapter.verify_claim(c)
            if verdict is None:
                continue
            if not verdict.true and c.confidence > 0.5:
                store.retract(c.statement, adapter.budget_spent())
