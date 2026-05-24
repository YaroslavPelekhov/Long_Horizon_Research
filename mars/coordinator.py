"""
Coordinator — non-LLM scheduler that wires Generator + Reflector +
MemorySelector + FutilityDetector + ClaimStore + Adapter.

The episode loop:

    while adapter.budget_left > 0 and not done:
        sub_goal = next_subgoal_from_adapter()
        if sub_goal is None: break
        for turn in range(MAX_TURNS_PER_SG):
            picked = MemorySelector.select(ClaimStore.active(), sub_goal_q, B)
            resp = Generator.propose(ctx_with_picked)
            for a in resp.actions:
                result = adapter.execute(a)
                history.append(...)
            for c in resp.claims:
                ClaimStore.assert/retract
            verdict = Reflector.review(last_actions, results, last_claims, picked)
            if verdict.verdict == "retract":  ClaimStore.retract(target)
            elif verdict == "revise":         ClaimStore.retract(target); next turn
            elif verdict == "require_evidence":  generator_feedback = reason; next turn
            elif verdict == "accept":         break
            FutilityDetector.check(...)
            if resp.halt:  return
        if subgoal_resolved: mark done; next subgoal

The Coordinator owns NO long-term state of its own — it threads ClaimStore +
adapter + the three agents and routes turns. All decisions are made by the
LLM agents or the heuristic FutilityDetector.

Ablation switches (for component-decomposition study, H3 in MARS_SPEC):
    use_reflector       — turn off Reflector dialogue (Generator + adapter only)
    use_memory_selector — turn off, give Generator ALL active claims (the OLS-v0.2
                          baseline behavior we're trying to fix)
    use_futility_detector — turn off FutilityDetector
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from ols.adapters.base import BudgetExhausted, ResearchEnvAdapter
from ols.core.abandon import FutilityDetector
from ols.core.agenda import AgendaController, SubGoalStatus    # used ONLY for tracking, not prioritization
from ols.core.claim_store import ClaimStore
from ols.core.types import AgendaItem, Claim

from mars.agents.generator import Generator, GenContext, GenResponse
from mars.agents.reflector import Reflector, ReflectorContext, ReflectorVerdict
from mars.agents.memory_selector import MemorySelector


@dataclass
class EpisodeReport:
    primary: float
    score_dict: dict
    n_turns: int
    n_actions: int
    n_claims_active: int
    n_claims_retracted: int
    n_subgoals_done: int
    n_subgoals_abandoned: int
    n_reflector_calls: int
    n_generator_calls: int
    verdict_counts: dict[str, int]
    axis6_regret: float
    wall_time_s: float
    ablations: dict[str, bool]
    seed: int | None = None
    history: list[dict] = field(default_factory=list)


@dataclass
class Coordinator:
    adapter: ResearchEnvAdapter
    generator: Generator
    reflector: Reflector
    memory_selector: MemorySelector
    futility: FutilityDetector = field(default_factory=lambda: FutilityDetector(theta_futile=0.30))
    # ablations
    use_reflector: bool = True
    use_memory_selector: bool = True
    use_futility_detector: bool = True
    # caps
    max_turns_per_subgoal: int = 4         # hard cap; prevents Reflector loops
    max_total_turns: int = 48
    seed: int | None = None
    verbose: bool = False
    # populated post-run
    last_store: ClaimStore | None = None

    def run_episode(self) -> EpisodeReport:
        t0 = time.time()
        adapter = self.adapter
        store = ClaimStore()
        self.last_store = store
        agenda = AgendaController()                # used as a simple tracker, NOT prioritizer
        history: list[dict] = []
        per_sg_history: dict[str, list[dict]] = {}
        verdict_counts = {v.value: 0 for v in ReflectorVerdict}
        total_turns = 0

        handle = adapter.handle()
        for sg, q in handle.subdomains:
            agenda.add(AgendaItem(sub_goal=sg, question=q,
                                  rationale="adapter-seeded"),
                       adapter.budget_spent())
            per_sg_history[sg] = []

        # iterate sub-goals in DECLARATION ORDER — no priority queue (we
        # established AgendaController's priority queue was harmful)
        sg_iter = list(handle.subdomains)
        sg_idx = 0

        try:
            while (adapter.budget_left() > 0
                   and total_turns < self.max_total_turns
                   and sg_idx < len(sg_iter)):

                cur_sg, cur_q = sg_iter[sg_idx]
                # skip if already abandoned/done
                if agenda.status.get(cur_sg) in (SubGoalStatus.DONE, SubGoalStatus.ABANDONED):
                    sg_idx += 1
                    continue
                agenda.promote(cur_sg, SubGoalStatus.PURSUE, adapter.budget_spent())

                turns_on_sg = 0
                pending_feedback: Optional[str] = None
                consecutive_req_evidence = 0   # cap Reflector loop pressure

                while turns_on_sg < self.max_turns_per_subgoal:
                    total_turns += 1
                    turns_on_sg += 1
                    cost_before = adapter.budget_spent()

                    # 1. MemorySelector
                    if self.use_memory_selector:
                        picked = self.memory_selector.select(
                            store.claims, cur_q,
                            handle.budget_total, current_sub_goal=cur_sg)
                    else:
                        # ablation: show ALL active claims (the OLS-v0.2 behavior)
                        picked = store.active()

                    picked_view = self.memory_selector.render_for_log(picked)

                    # 2. Generator
                    g_ctx = GenContext(
                        sub_goal=cur_sg,
                        sub_goal_question=cur_q,
                        env_description=handle.description,
                        available_actions=handle.actions,
                        selected_claims=picked,
                        history_this_subgoal=list(per_sg_history[cur_sg][-5:]),
                        budget_remaining=adapter.budget_left(),
                        budget_total=handle.budget_total,
                        reflector_feedback=pending_feedback,
                    )
                    g_resp = self.generator.propose(g_ctx)
                    pending_feedback = None      # consumed

                    # 3. Execute actions
                    new_eids: list[int] = []
                    action_results: list[dict] = []
                    for a in g_resp.actions:
                        try:
                            result = adapter.execute(a["action"], a["args"])
                        except BudgetExhausted:
                            raise
                        new_eids.append(result.eid)
                        action_results.append({
                            "eid": result.eid,
                            "action": a["action"],
                            "args": a["args"],
                            "cost": result.cost,
                            "summary": result.summary,
                        })
                        per_sg_history[cur_sg].append({
                            "eid": result.eid,
                            "action": a["action"],
                            "args": a["args"],
                            "summary": result.summary,
                        })

                    # 4. Claim deltas
                    asserted_idxs: list[int] = []
                    last_asserted: list[dict] = []
                    for c in g_resp.claims:
                        if c["op"] == "retract":
                            store.retract(c["statement"], adapter.budget_spent())
                        else:
                            idx = store.assert_claim(
                                statement=c["statement"],
                                confidence=c["confidence"],
                                provenance=new_eids or [],
                                budget=adapter.budget_spent(),
                                sub_goal=cur_sg,
                            )
                            asserted_idxs.append(idx)
                            last_asserted.append(c)
                            agenda.record_claim(cur_sg)
                    agenda.record_spend(cur_sg,
                                        max(0.0, adapter.budget_spent() - cost_before))

                    # 5. Reflector — fire ONLY when Generator emitted a claim
                    # (Reflector's purpose is to critique claim assertions,
                    # not action exploration; observe-only turns pass through
                    # so Generator can accumulate evidence before claiming).
                    if self.use_reflector and g_resp.claims:
                        r_ctx = ReflectorContext(
                            sub_goal=cur_sg,
                            sub_goal_question=cur_q,
                            last_actions=g_resp.actions,
                            last_action_results=action_results,
                            last_claims=last_asserted,
                            selected_claims_view=picked_view,
                            budget_remaining=adapter.budget_left(),
                        )
                        r_resp = self.reflector.review(r_ctx)
                        verdict_counts[r_resp.verdict.value] = (
                            verdict_counts.get(r_resp.verdict.value, 0) + 1
                        )
                        if r_resp.verdict == ReflectorVerdict.RETRACT and r_resp.target_claim:
                            store.retract(r_resp.target_claim, adapter.budget_spent())
                            pending_feedback = (
                                f"Reflector RETRACT on '{r_resp.target_claim[:120]}': "
                                f"{r_resp.reason}"
                            )
                        elif r_resp.verdict == ReflectorVerdict.REVISE and r_resp.target_claim:
                            store.retract(r_resp.target_claim, adapter.budget_spent())
                            pending_feedback = (
                                f"Reflector REVISE on '{r_resp.target_claim[:120]}': "
                                f"{r_resp.reason}. Re-emit a sharper version."
                            )
                        elif r_resp.verdict == ReflectorVerdict.REQUIRE_EVIDENCE:
                            consecutive_req_evidence += 1
                            if consecutive_req_evidence >= 2:
                                # 2-й подряд → force ACCEPT to break loops.
                                # Convert verdict for the log, drop feedback.
                                verdict_counts[ReflectorVerdict.REQUIRE_EVIDENCE.value] -= 1
                                verdict_counts[ReflectorVerdict.ACCEPT.value] = (
                                    verdict_counts.get(ReflectorVerdict.ACCEPT.value, 0) + 1
                                )
                                r_resp.verdict = ReflectorVerdict.ACCEPT
                                pending_feedback = None
                            else:
                                pending_feedback = (
                                    f"Reflector REQUIRE_EVIDENCE: {r_resp.reason}. "
                                    "Run one verifying action before advancing."
                                )
                        # ACCEPT → no feedback; continue
                        if r_resp.verdict != ReflectorVerdict.REQUIRE_EVIDENCE:
                            consecutive_req_evidence = 0
                    else:
                        r_resp = None

                    # 6. Futility
                    if self.use_futility_detector:
                        for fg in self.futility.check(
                            agenda, handle.budget_total,
                            adapter.budget_left(), adapter.budget_spent(),
                        ):
                            agenda.abandon(fg, adapter.budget_spent())

                    history.append({
                        "t": total_turns,
                        "sub_goal": cur_sg,
                        "turn_on_sg": turns_on_sg,
                        "n_actions": len(g_resp.actions),
                        "n_claims": len(g_resp.claims),
                        "verdict": (r_resp.verdict.value if r_resp else None),
                        "picked_view": picked_view[:200],
                    })

                    if self.verbose:
                        v = r_resp.verdict.value if r_resp else "—"
                        print(f"[MARS] t={total_turns} sg={cur_sg} a={len(g_resp.actions)} "
                              f"c={len(g_resp.claims)} v={v}")

                    if g_resp.halt:
                        agenda.mark_done(cur_sg, adapter.budget_spent())
                        raise StopIteration

                    if (r_resp and r_resp.verdict == ReflectorVerdict.ACCEPT
                            and g_resp.advance_subgoal):
                        agenda.mark_done(cur_sg, adapter.budget_spent())
                        break
                    if (not self.use_reflector) and g_resp.advance_subgoal:
                        # No Reflector — Generator's advance decision is final
                        agenda.mark_done(cur_sg, adapter.budget_spent())
                        break
                    if agenda.status.get(cur_sg) == SubGoalStatus.ABANDONED:
                        break

                # exhausted turns on this sub-goal without explicit done:
                if agenda.status.get(cur_sg) not in (SubGoalStatus.DONE, SubGoalStatus.ABANDONED):
                    agenda.mark_done(cur_sg, adapter.budget_spent())
                sg_idx += 1

        except (BudgetExhausted, StopIteration):
            pass

        active = store.active()
        score = adapter.score_episode(active, final_artifact=None)
        primary = score.get("primary", score.get("RPS", score.get("HMS", 0.0)))

        n_done = sum(1 for s in agenda.status.values() if s == SubGoalStatus.DONE)
        n_aband = sum(1 for s in agenda.status.values() if s == SubGoalStatus.ABANDONED)

        return EpisodeReport(
            primary=primary,
            score_dict=score,
            n_turns=total_turns,
            n_actions=sum(h["n_actions"] for h in history),
            n_claims_active=len(active),
            n_claims_retracted=len([c for c in store.claims if c.status == "retracted"]),
            n_subgoals_done=n_done,
            n_subgoals_abandoned=n_aband,
            n_reflector_calls=self.reflector.n_calls,
            n_generator_calls=self.generator.n_calls,
            verdict_counts=verdict_counts,
            axis6_regret=self.futility.regret(agenda, handle.budget_total),
            wall_time_s=time.time() - t0,
            ablations={
                "use_reflector": self.use_reflector,
                "use_memory_selector": self.use_memory_selector,
                "use_futility_detector": self.use_futility_detector,
            },
            seed=self.seed,
            history=history,
        )
