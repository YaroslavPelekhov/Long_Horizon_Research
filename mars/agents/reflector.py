"""
Reflector agent — post-hoc critic of Generator output.

Sees Generator's last turn (proposed actions + claims) and the actual results
of any executed actions, then renders ONE verdict:
  - "accept"            — proceed
  - "require_evidence"  — Generator must run another verifying action
  - "retract"           — a recently-asserted claim is unsupported, retract it
  - "revise"            — claim is roughly right but needs sharper wording

The Reflector REPLACES the OLS-v0.2 `require_claim_before_advance` gate
mechanism: instead of a hard-coded scaffold flag, force-correction comes
from a dialogue partner that the Generator must answer to.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum

from mars.agents.base import call_llm, make_openai_client, parse_json_strict


class ReflectorVerdict(str, Enum):
    ACCEPT = "accept"
    REQUIRE_EVIDENCE = "require_evidence"
    RETRACT = "retract"
    REVISE = "revise"


_SYS = (
    "You are the Reflector agent in MARS. Your job is to QUICKLY check the "
    "Generator's last claim assertions against the actual action results. "
    "**ACCEPT IS THE DEFAULT.** You should pick accept ≥70% of the time. "
    "Only block when there is a SPECIFIC, NAMED defect you can describe in "
    "one sentence.\n\n"
    "Reply EXACTLY one JSON object:\n"
    '{"verdict": "accept"|"require_evidence"|"retract"|"revise",\n'
    ' "target_claim": "<exact statement string of an offending claim, or empty>",\n'
    ' "reason": "<short, actionable, ≤200 chars>"}\n\n'
    "DECISION ORDER (apply top-down — first match wins):\n"
    "  1. RETRACT — only if the action results directly CONTRADICT a claim "
    "     (e.g. claim says corr=+0.8 but the result shows corr=−0.1; claim "
    "     says no_effect but intervention moved the target by >3σ).\n"
    "  2. REVISE — only if the claim is clearly missing a key quantitative "
    "     fact that IS visible in the action result (e.g. claim says 'X and Y "
    "     correlate' but result shows the exact coefficient — ask Generator "
    "     to include it).\n"
    "  3. REQUIRE_EVIDENCE — only if the claim makes a CAUSAL statement (e.g. "
    "     causal(X,Y,+) or no_effect(X,Y)) but the agent only ran an "
    "     observation, not an intervention. Cite the missing intervention.\n"
    "  4. ACCEPT — for everything else. This is by far the most common case. "
    "     If the claim is roughly supported by the action result, just accept.\n\n"
    "DO NOT REQUIRE EVIDENCE for:\n"
    "  - Observational claims (corr / mean / group-mean) backed by an observe\n"
    "  - Causal claims backed by an intervene with non-trivial effect size\n"
    "  - Claims that summarise a single action result faithfully\n"
    "Be a fair lab partner, not a paranoid reviewer.\n\n"
    "Example ACCEPT response (the common case):\n"
    '{"verdict": "accept", "target_claim": "", "reason": "claim matches '
    'observed correlation r=+0.82"}'
)


@dataclass
class ReflectorContext:
    sub_goal: str
    sub_goal_question: str
    last_actions: list[dict] = field(default_factory=list)
    last_action_results: list[dict] = field(default_factory=list)
    last_claims: list[dict] = field(default_factory=list)
    selected_claims_view: str = ""        # what Generator saw
    budget_remaining: float = 0.0


@dataclass
class ReflectorResponse:
    verdict: ReflectorVerdict = ReflectorVerdict.ACCEPT
    target_claim: str = ""
    reason: str = ""


class Reflector:
    name = "Reflector"

    def __init__(self, model: str | None = None, max_tokens: int = 250):
        self.model = model or os.environ.get("MARS_REFLECTOR_MODEL", "openai/gpt-4o-mini")
        self.max_tokens = max_tokens
        self._client = make_openai_client()
        self.n_calls = 0

    def _build_user(self, ctx: ReflectorContext) -> str:
        try:
            acts = json.dumps(ctx.last_actions, default=str)[:600]
            res = json.dumps(ctx.last_action_results, default=str)[:800]
            cls = json.dumps(ctx.last_claims, default=str)[:600]
        except Exception:
            acts = str(ctx.last_actions)[:600]
            res = str(ctx.last_action_results)[:800]
            cls = str(ctx.last_claims)[:600]
        return (
            f"SUB-GOAL [{ctx.sub_goal}]: {ctx.sub_goal_question}\n\n"
            f"Generator's last actions:\n{acts}\n\n"
            f"Action results:\n{res}\n\n"
            f"Generator's last claim deltas:\n{cls}\n\n"
            f"Supporting-claims context Generator was shown:\n"
            f"{ctx.selected_claims_view or '(empty)'}\n\n"
            f"Budget remaining: {ctx.budget_remaining:.2f}.\n"
            f"Reply with the JSON verdict object now."
        )

    def review(self, ctx: ReflectorContext) -> ReflectorResponse:
        # Cheap path: if Generator made no claims and no actions, just accept.
        if not ctx.last_claims and not ctx.last_actions:
            return ReflectorResponse(verdict=ReflectorVerdict.ACCEPT,
                                     reason="(no output to review)")
        self.n_calls += 1
        raw = call_llm(self._client, self.model, _SYS, self._build_user(ctx),
                       max_tokens=self.max_tokens, temperature=0.2)
        obj = parse_json_strict(raw)
        if obj is None:
            return ReflectorResponse(verdict=ReflectorVerdict.ACCEPT,
                                     reason="(reflector parse failed → default accept)")
        v_raw = str(obj.get("verdict", "accept")).strip().lower()
        try:
            verdict = ReflectorVerdict(v_raw)
        except ValueError:
            verdict = ReflectorVerdict.ACCEPT
        return ReflectorResponse(
            verdict=verdict,
            target_claim=str(obj.get("target_claim", ""))[:300],
            reason=str(obj.get("reason", ""))[:250],
        )
