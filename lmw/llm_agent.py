"""
LLM agents on the construct-valid LMW harness.

  BaselineLLM : thin loop, only the recent transcript in context (no scaffold).
  StrongLLM   : explicit scaffolding for the three outer-loop axes --
                * axis 2: a STRUCTURED, persistent theory view (established
                  claims + per-cluster status) fed every turn, plus an explicit
                  `synthesize` action prompting it to derive confounded(a,b,Z)
                  from remembered no_effect + true-cause sub-claims;
                * axis 1: an explicit agenda it sets/revises;
                * axis 6: an explicit `abandon` action + budget-aware rule.
                memory=False ablates ONLY the structured theory/agenda view
                (it then sees just the last transcript line) -> isolates whether
                a REAL agent's RPS depends on the persistent-theory scaffold,
                i.e. whether the metric is sensitive to axis 2 for a live agent.

Provenance is tracked by the wrapper (a real system's job): honest claims pass
the scorer's audit; unsupported ones are correctly penalised.

Cheap by construction: gpt-4o-mini, hard call cap. OpenAI SDK -> OpenRouter
(OPENAI_BASE_URL / OPENAI_API_KEY from env).
"""

from __future__ import annotations

import json

from claims import AgendaItem, ClaimStore
from world import BudgetExhausted, World

GRAMMAR = ("causal(A,B,+) | causal(A,B,-) | no_effect(A,B) | "
           "confounded(A,B,Z) | mean(V)~const")

SYS = (
    "You are an autonomous research agent. Maximise TRUE, high-value causal "
    "facts within a tight experiment budget. observe() shows only ASSOCIATION; "
    "do-interventions reveal CAUSATION. A pair can be strongly correlated yet "
    "NON-causal due to a hidden common cause -> that is confounded(A,B,Z), and "
    "you may assert it only after you have established BOTH no_effect(A,B) (an "
    "intervention on A does not move B) AND some causal(T,B) for another var T. "
    "Some clusters look promising but every intervention is null: probe briefly "
    "then ABANDON to save budget. Claim only what your experiments support. "
    f"Grammar (exact): {GRAMMAR}. Reply with EXACTLY ONE JSON object, no prose."
)

ACTIONS_BASE = ('{"action":"observe","cluster":C} | '
                '{"action":"intervene","cause":V,"target":V} | '
                '{"action":"claim","statement":S} | '
                '{"action":"abandon","cluster":C} | {"action":"stop"}')
ACTIONS_STRONG = ACTIONS_BASE.replace(
    '{"action":"stop"}',
    '{"action":"set_agenda","order":[C,...]} | '
    '{"action":"synthesize"} | {"action":"stop"}')


class _LLMBase:
    def __init__(self, model="openai/gpt-4o-mini", max_calls=36):
        self.model, self.max_calls, self.calls = model, max_calls, 0
        from openai import OpenAI
        self._client = OpenAI()

    def _support(self, stmt: str, world: World) -> list[int]:
        log = world.log
        def ev(c, t):
            return [e.eid for e in log if e.kind == "intervene"
                    and c in e.interventions and t in e.measured]
        def ob(v):
            return [e.eid for e in log if e.kind == "observe" and v in e.measured]
        s = stmt.replace(" ", "")
        if s.startswith("causal(") or s.startswith("no_effect("):
            p = s[s.index("(") + 1:s.index(")")].split(",")
            return ev(p[0], p[1])[-2:]
        if s.startswith("confounded("):
            a, b, _ = s[s.index("(") + 1:s.index(")")].split(",")
            return (ob(a)[-1:] + ev(a, b)[-2:]) or ev(a, b)
        if s.startswith("mean("):
            return ob(s[s.index("(") + 1:s.index(")~")])[-1:]
        return []

    def _ask(self, user: str) -> dict | None:
        self.calls += 1
        try:
            r = self._client.chat.completions.create(
                model=self.model, temperature=0.2, max_tokens=140,
                messages=[{"role": "system", "content": SYS},
                          {"role": "user", "content": user}])
            t = r.choices[0].message.content.strip()
            if "```" in t:
                t = t.split("```")[1].lstrip("json").strip()
            return json.loads(t[t.find("{"):t.rfind("}") + 1])
        except Exception as e:
            print(f"[LLM] {e}")
            return None

    # shared action execution; returns transcript line
    def _do(self, act, world, store, clusters) -> str:
        a = act.get("action")
        if a == "observe":
            vs = clusters.get(act.get("cluster"), [])
            if not vs:
                return "observe: bad cluster"
            e = world.observe(vs, n=8)
            cr = {f"{vs[i]}~{vs[j]}": round(e.corr(vs[i], vs[j]), 2)
                  for i in range(len(vs)) for j in range(i + 1, len(vs))}
            return f"observe {act['cluster']}: corr={cr}"
        if a == "intervene":
            c, t = act.get("cause"), act.get("target")
            hi = world.intervene(c, +2.0, [t], n=6)
            lo = world.intervene(c, -2.0, [t], n=6)
            d = sum(s[t] for s in hi.samples) / 6 - sum(s[t] for s in lo.samples) / 6
            return f"do({c})->{t}: delta={round(d,2)} (|d|>~1 => causal)"
        if a == "claim":
            st = str(act.get("statement", "")).replace(" ", "")
            store.assert_claim(st, 0.8, self._support(st, world),
                               world.budget_spent)
            return f"claim {st}"
        if a == "abandon":
            cn = act.get("cluster")
            store.set_agenda([x for x in store.agenda if x.subdomain != cn],
                             world.budget_spent)
            return f"abandon {cn}"
        if a == "set_agenda":
            order = [c for c in act.get("order", []) if c in clusters]
            store.set_agenda([AgendaItem(c, "pursue", "llm") for c in order],
                             world.budget_spent)
            return f"agenda={order}"
        return f"noop({a})"


class BaselineLLM(_LLMBase):
    name = "LLM-base"

    def run(self, world: World, store: ClaimStore) -> None:
        clusters = dict(world._scm.structure.subdomains)
        tr: list[str] = []
        while world.budget_left > 1.0 and self.calls < self.max_calls:
            st = {"budget_left": round(world.budget_left, 1),
                  "clusters": clusters, "recent": tr[-5:]}
            act = self._ask(f"STATE:{json.dumps(st)}\nACTIONS:{ACTIONS_BASE}\n"
                            "Next single JSON action:")
            if not act or act.get("action") == "stop":
                break
            try:
                tr.append(self._do(act, world, store, clusters))
            except BudgetExhausted:
                break


class StrongLLM(_LLMBase):
    name = "LLM-strong"

    def __init__(self, memory=True, **kw):
        super().__init__(**kw)
        self.memory = memory

    def run(self, world: World, store: ClaimStore) -> None:
        clusters = dict(world._scm.structure.subdomains)
        status = {c: "unexplored" for c in clusters}
        tr: list[str] = []
        store.set_agenda([AgendaItem(c, "init", "llm") for c in clusters],
                         world.budget_spent)

        while world.budget_left > 1.0 and self.calls < self.max_calls:
            if self.memory:
                theory = {
                    "established": sorted({c.statement for c in store.active()}),
                    "cluster_status": status,
                    "agenda": [a.subdomain for a in store.agenda],
                    "budget_left": round(world.budget_left, 1),
                    "clusters": clusters,
                    "recent": tr[-4:],
                }
                hint = ("Use the theory. If you have no_effect(A,B) AND a "
                        "causal(T,B), call synthesize or claim "
                        "confounded(A,B,Z). Abandon inert clusters.")
                acts = ACTIONS_STRONG
            else:
                theory = {"budget_left": round(world.budget_left, 1),
                          "clusters": clusters, "recent": tr[-1:]}
                hint = ""
                acts = ACTIONS_STRONG
            act = self._ask(f"THEORY:{json.dumps(theory)}\n{hint}\n"
                            f"ACTIONS:{acts}\nNext single JSON action:")
            if not act or act.get("action") == "stop":
                break
            try:
                if act.get("action") == "synthesize":
                    tr.append(self._synthesize(world, store))
                else:
                    line = self._do(act, world, store, clusters)
                    tr.append(line)
                    cl = act.get("cluster")
                    if cl in status:
                        if act.get("action") == "abandon":
                            status[cl] = "abandoned"
                        elif "delta=" in line:
                            status[cl] = "probed"
            except BudgetExhausted:
                break

    def _synthesize(self, world: World, store: ClaimStore) -> str:
        """Explicit axis-2 step: derive confounded from remembered sub-claims."""
        act = {c.statement for c in store.active()}
        made = []
        noeff = [s for s in act if s.startswith("no_effect(")]
        for ne in noeff:
            a, b = ne[len("no_effect("):-1].split(",")
            has_cause = any(s.startswith("causal(") and s.split(",")[1] == b
                            and not s.startswith(f"causal({a},") for s in act)
            cf = f"confounded({a},{b},Z)"
            if has_cause and cf not in act:
                store.assert_claim(cf, 0.85, self._support(cf, world),
                                   world.budget_spent)
                made.append(cf)
        return f"synthesize -> {made or 'nothing new'}"
