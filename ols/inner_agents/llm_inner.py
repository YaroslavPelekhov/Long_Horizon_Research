"""
OLSInnerLLM — a domain-agnostic inner LLM agent driven by an OLSScaffold.

Same class drives LMWAdapter, DiscoveryBenchAdapter, and any future adapter
that implements ResearchEnvAdapter. The agent sees ONLY what the scaffold
exposes via AgentContext — env description, the current sub-goal, the
available actions (with arg schemas), theory-state view, agenda view, and
budget remaining. There are NO benchmark-specific priors hard-coded in here.

The same agent is used for both treatment and baseline runs (the difference
is which OLS ablation flags are set — see ols/scaffold.py and the runners).
This is the cleanest A/B: the comparison isolates the three OLS modules
(ClaimStore, Agenda, Futility) and not differences in prompting style.
"""

from __future__ import annotations

import json
import os

from ols.inner_agents.base import (
    ActionRequest,
    AgentContext,
    AgentResponse,
    ClaimDelta,
)


_SYS = (
    "You are a research agent working under an outer-loop scaffold. The "
    "scaffold gives you ONE sub-goal at a time. For each sub-goal, decide on "
    "one or more actions to execute (within the listed action set) and "
    "optionally assert/retract claims based on action outcomes. "
    "You may take advantage of any THEORY STATE shown to you — earlier claims "
    "asserted under other sub-goals — to avoid redundant work and to "
    "synthesise compound claims. "
    "Reply with EXACTLY one JSON object, no prose, of the form: "
    '{"rationale": "...", "actions": [{"action": ACTION_NAME, "args": {...}}], '
    '"claims": [{"op": "assert"|"retract", "statement": STR, "confidence": 0.0..1.0}], '
    '"advance_subgoal": true|false, "halt": true|false}. '
    "Hard rules: at most 2 actions per turn; only assert when action history "
    "supports it; set advance_subgoal true when you believe THIS sub-goal "
    "is resolved (the scaffold will move on). Set halt only to end the entire "
    "episode."
)


class OLSInnerLLM:
    name = "OLSInnerLLM"

    def __init__(
        self,
        model: str = None,
        max_tokens: int = 700,
        max_actions_per_turn: int = 2,
    ):
        from openai import OpenAI                                # local import
        self.model = model or os.environ.get("OLS_INNER_MODEL", "openai/gpt-4o-mini")
        self.max_tokens = max_tokens
        self.max_actions_per_turn = max_actions_per_turn
        # Auto-detect OpenRouter from key prefix; honor explicit OPENAI_BASE_URL.
        key = os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get("OPENAI_BASE_URL")
        if not base_url and key.startswith("sk-or-"):
            base_url = "https://openrouter.ai/api/v1"
        if base_url:
            self._client = OpenAI(base_url=base_url, api_key=key)
        else:
            self._client = OpenAI()
        self._n_calls = 0

    # -- prompt construction --------------------------------------------------

    def _render_actions(self, ctx: AgentContext) -> str:
        out = []
        for a in ctx.available_actions:
            out.append(
                f'  - {a.name}(args: {json.dumps(a.arg_schema)})  '
                f'-- est_cost≈{a.cost_estimate:.2f} -- {a.description}'
            )
        return "\n".join(out)

    def _render_history(self, ctx: AgentContext) -> str:
        if not ctx.history_for_this_subgoal:
            return "(no prior actions on this sub-goal yet)"
        lines = []
        for h in ctx.history_for_this_subgoal[-6:]:
            summ = h.get("summary", {})
            # Compact summary: keep keys + scalar/dict-of-scalars values
            try:
                summ_s = json.dumps(summ, default=str)[:400]
            except Exception:
                summ_s = str(summ)[:400]
            lines.append(f"  - eid={h.get('eid')} {h.get('action')}({json.dumps(h.get('args', {}))}) "
                         f"→ {summ_s}")
        return "Prior actions on this sub-goal:\n" + "\n".join(lines)

    def _build_user(self, ctx: AgentContext) -> str:
        return (
            f"ENVIRONMENT:\n{ctx.env_description}\n\n"
            f"AVAILABLE ACTIONS:\n{self._render_actions(ctx)}\n\n"
            f"CURRENT SUB-GOAL: [{ctx.sub_goal}]\n"
            f"  Question: {ctx.sub_goal_question}\n\n"
            f"{ctx.agenda_view}\n\n"
            f"{ctx.theory_state_view}\n\n"
            f"{self._render_history(ctx)}\n\n"
            f"Budget remaining: {ctx.budget_remaining:.2f} / {ctx.budget_total:.2f}.\n"
            f"Reply with the JSON object now."
        )

    # -- call -----------------------------------------------------------------

    def _call(self, sys: str, user: str) -> str:
        self._n_calls += 1
        # First attempt: with response_format=json_object (cleanest path on
        # native OpenAI). If it throws OR returns empty content (OpenRouter
        # sometimes silently rejects response_format on mini), retry without.
        for use_rf in (True, False):
            try:
                kw = dict(
                    model=self.model,
                    messages=[{"role": "system", "content": sys},
                              {"role": "user", "content": user}],
                    temperature=0.4,
                    max_tokens=self.max_tokens,
                )
                if use_rf:
                    kw["response_format"] = {"type": "json_object"}
                r = self._client.chat.completions.create(**kw)
                content = (r.choices[0].message.content or "").strip()
                if content:
                    return content
            except Exception:
                continue
        return ""

    # -- parse ----------------------------------------------------------------

    def _parse(self, raw: str, ctx: AgentContext) -> AgentResponse:
        if not raw:
            return AgentResponse(advance_subgoal=True, rationale="(empty completion)")

        # tolerate fenced code blocks
        s = raw.strip()
        if s.startswith("```"):
            s = s.strip("` \n")
            if s.startswith("json"):
                s = s[4:]
        try:
            obj = json.loads(s)
        except Exception:
            # last-ditch: find first {...} substring
            i, j = s.find("{"), s.rfind("}")
            if 0 <= i < j:
                try:
                    obj = json.loads(s[i:j + 1])
                except Exception:
                    return AgentResponse(advance_subgoal=True,
                                         rationale="(unparseable JSON)")
            else:
                return AgentResponse(advance_subgoal=True,
                                     rationale="(unparseable JSON)")

        action_names = {a.name for a in ctx.available_actions}
        actions: list[ActionRequest] = []
        for a in (obj.get("actions") or [])[: self.max_actions_per_turn]:
            if not isinstance(a, dict):
                continue
            name = str(a.get("action", "")).strip()
            args = a.get("args") or {}
            if name in action_names and isinstance(args, dict):
                actions.append(ActionRequest(action=name, args=args))

        claims: list[ClaimDelta] = []
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
            claims.append(ClaimDelta(
                op=op, statement=stmt,
                confidence=max(0.0, min(1.0, conf)),
            ))

        return AgentResponse(
            actions=actions,
            claims=claims,
            advance_subgoal=bool(obj.get("advance_subgoal", False)),
            halt=bool(obj.get("halt", False)),
            rationale=str(obj.get("rationale", ""))[:400],
        )

    # -- protocol -------------------------------------------------------------

    def propose(self, ctx: AgentContext) -> AgentResponse:
        raw = self._call(_SYS, self._build_user(ctx))
        return self._parse(raw, ctx)
