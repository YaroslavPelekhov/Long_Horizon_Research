"""
Generator agent — proposes 1-2 actions per turn + zero/one claim.

Narrow contract:
  Input: current sub-goal, available actions (with arg schemas), TOP-K
  selected claims (NOT the whole ClaimStore), recent history of THIS sub-goal,
  budget remaining, and optionally a last-Reflector-feedback note.
  Output: JSON object with actions, claims, advance_subgoal, halt.

Key differences from ols.inner_agents.OLSInnerLLM:
  - Sees only top-k claims (from MemorySelector), not full ClaimStore
  - No "agenda view" (we dropped AgendaController)
  - Receives an explicit reflector_feedback field when prior turn was rejected
  - Always tries to emit a claim when an action returned a concrete result
    (forced by Reflector dialogue, not by scaffold gate)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from mars.agents.base import call_llm, make_openai_client, parse_json_strict
from ols.core.types import ActionSpec, Claim                  # reuse types


_SYS = (
    "You are the Generator agent in MARS, a multi-agent research system. "
    "On each turn you receive ONE sub-goal and a small set of pre-selected "
    "supporting claims (chosen by the MemorySelector — do NOT ask for more). "
    "Propose up to 2 actions to execute next, and optionally ASSERT a claim "
    "describing a concrete finding from a prior action result. A separate "
    "Reflector agent will critique your output; if your previous turn was "
    "rejected, you will see reflector_feedback explaining what to fix.\n\n"
    "Reply EXACTLY one JSON object, no prose:\n"
    '{"rationale": "...",\n'
    ' "actions": [{"action": ACTION_NAME, "args": {...}}, ...],\n'
    ' "claims": [{"op": "assert"|"retract", "statement": STR, "confidence": 0.0..1.0}],\n'
    ' "advance_subgoal": true|false,\n'
    ' "halt": true|false}\n\n'
    "RULES:\n"
    "  R1. Max 2 actions per turn.\n"
    "  R2. After an action returns a concrete number / correlation / effect / "
    "category, ASSERT a corresponding claim within the next turn. Do NOT skip "
    "to terminal actions (submit_*) without asserting intermediate findings — "
    "the Reflector will reject that.\n"
    "  R3. Claim statements are short, opaque, domain-natural facts (e.g. "
    "'corr(MBL_evol, speciation_rate) ≈ +0.82', 'no_effect(X2,X3)', "
    "'group_mean(income,region=north) > group_mean(income,region=south)').\n"
    "  R4. Set advance_subgoal=true only when current sub-goal is resolved "
    "AND you have at least one asserted claim under it.\n"
    "  R5. Set halt=true only to end the whole episode (rare)."
)


@dataclass
class GenContext:
    sub_goal: str
    sub_goal_question: str
    env_description: str
    available_actions: list[ActionSpec]
    selected_claims: list[Claim] = field(default_factory=list)   # MemorySelector top-k
    history_this_subgoal: list[dict] = field(default_factory=list)
    budget_remaining: float = 0.0
    budget_total: float = 0.0
    reflector_feedback: str | None = None                        # set by Coordinator


@dataclass
class GenResponse:
    actions: list[dict] = field(default_factory=list)            # {"action": str, "args": dict}
    claims: list[dict] = field(default_factory=list)             # {"op": ..., "statement": ..., "confidence": ...}
    advance_subgoal: bool = False
    halt: bool = False
    rationale: str = ""


class Generator:
    name = "Generator"

    def __init__(self, model: str | None = None, max_tokens: int = 700,
                 max_actions_per_turn: int = 2):
        self.model = model or os.environ.get("MARS_GENERATOR_MODEL", "openai/gpt-4o-mini")
        self.max_tokens = max_tokens
        self.max_actions_per_turn = max_actions_per_turn
        self._client = make_openai_client()
        self.n_calls = 0

    # -- prompt ----

    def _render_actions(self, ctx: GenContext) -> str:
        out = []
        for a in ctx.available_actions:
            out.append(
                f'  - {a.name}(args: {json.dumps(a.arg_schema)})  '
                f'-- est_cost≈{a.cost_estimate:.2f} -- {a.description}'
            )
        return "\n".join(out)

    def _render_claims(self, ctx: GenContext) -> str:
        if not ctx.selected_claims:
            return "(no supporting claims selected — fresh sub-goal)"
        lines = [
            f"  - [{c.confidence:.2f}] {c.statement}"
            for c in ctx.selected_claims
        ]
        return f"Top-{len(lines)} supporting claims (selected by MemorySelector):\n" + "\n".join(lines)

    def _render_history(self, ctx: GenContext) -> str:
        if not ctx.history_this_subgoal:
            return "(no prior actions on this sub-goal)"
        lines = []
        for h in ctx.history_this_subgoal[-5:]:
            summ = h.get("summary", {})
            try:
                s = json.dumps(summ, default=str)[:380]
            except Exception:
                s = str(summ)[:380]
            lines.append(
                f"  - eid={h.get('eid')} {h.get('action')}"
                f"({json.dumps(h.get('args', {}))}) → {s}"
            )
        return "Recent actions on this sub-goal:\n" + "\n".join(lines)

    def _build_user(self, ctx: GenContext) -> str:
        reflector_block = ""
        if ctx.reflector_feedback:
            reflector_block = (
                "\nREFLECTOR FEEDBACK (your previous output was rejected):\n"
                f"  {ctx.reflector_feedback}\n"
                "Apply this feedback before issuing your next turn.\n"
            )
        return (
            f"ENVIRONMENT:\n{ctx.env_description}\n\n"
            f"AVAILABLE ACTIONS:\n{self._render_actions(ctx)}\n\n"
            f"CURRENT SUB-GOAL [{ctx.sub_goal}]: {ctx.sub_goal_question}\n\n"
            f"{self._render_claims(ctx)}\n\n"
            f"{self._render_history(ctx)}\n"
            f"{reflector_block}\n"
            f"Budget remaining: {ctx.budget_remaining:.2f} / {ctx.budget_total:.2f}.\n"
            f"Reply with the JSON object now."
        )

    # -- call & parse ----

    def propose(self, ctx: GenContext) -> GenResponse:
        self.n_calls += 1
        raw = call_llm(self._client, self.model, _SYS, self._build_user(ctx),
                       max_tokens=self.max_tokens)
        obj = parse_json_strict(raw)
        if obj is None:
            # Empty completion → end-of-sub-goal with no actions (Coordinator
            # will move on). Better than infinite-loop.
            return GenResponse(advance_subgoal=True, rationale="(unparseable)")

        # action_names guard
        avail = {a.name for a in ctx.available_actions}
        actions = []
        for a in (obj.get("actions") or [])[: self.max_actions_per_turn]:
            if not isinstance(a, dict):
                continue
            name = str(a.get("action", "")).strip()
            args = a.get("args") or {}
            if name in avail and isinstance(args, dict):
                actions.append({"action": name, "args": args})

        claims = []
        for c in (obj.get("claims") or []):
            if not isinstance(c, dict):
                continue
            stmt = str(c.get("statement", "")).strip()
            if not stmt:
                continue
            op = str(c.get("op", "assert")).strip()
            if op not in ("assert", "retract"):
                op = "assert"
            try:
                conf = float(c.get("confidence", 0.8))
            except Exception:
                conf = 0.8
            claims.append({
                "op": op, "statement": stmt,
                "confidence": max(0.0, min(1.0, conf)),
            })

        return GenResponse(
            actions=actions,
            claims=claims,
            advance_subgoal=bool(obj.get("advance_subgoal", False)),
            halt=bool(obj.get("halt", False)),
            rationale=str(obj.get("rationale", ""))[:400],
        )
