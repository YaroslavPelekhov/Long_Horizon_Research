"""
AutoDiscovery-algorithm adapter agent.

Faithful re-implementation of the AutoDiscovery algorithm (Agarwal et al.,
NeurIPS 2025): open-ended hypothesis discovery via Bayesian-surprise reward +
MCTS with progressive widening + LLM belief elicitation, on the LMW `World`
action surface. NOT a wrap of the original codebase. See
`AUTODISCOVERY_ADAPTER_v0.md` for the rationale and contract.

Reads only the allowed surface (`world._scm.structure.subdomains`,
`world.observe/intervene`, budget); never touches Oracle / Structure.roles /
spurious_pair / true_cause / fake_const / Edge.coeff. Emits claims in the
strict grammar with honest provenance.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field

from claims import ClaimStore
from world import BudgetExhausted, World

try:
    from scipy.special import betaln as _betaln, digamma as _digamma
except ImportError:  # cheap stdlib fallback
    def _betaln(a, b): return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    def _digamma(x): return math.log(x) - 1.0 / (2.0 * x)


SYS = (
    "You are an autonomous research agent operating in a small synthetic "
    "world with opaque clusters of variables. You can observe and intervene; "
    "score high by discovering TRUE high-value causal/null/confounding facts "
    "while AVOIDING confidently FALSE claims (an integrity penalty applies). "
    "Some clusters look promising but are inert -- probe briefly then abandon. "
    "Strict claim grammar (the ONLY allowed statement forms): "
    "causal(A,B,+) | causal(A,B,-) | no_effect(A,B) | confounded(A,B,Z) | "
    "mean(V)~const. Reply with EXACTLY one JSON object, no prose."
)

GRAMMAR_RE = re.compile(
    r"^(causal\([^,()]+,[^,()]+,[+-]\)|no_effect\([^,()]+,[^,()]+\)|"
    r"confounded\([^,()]+,[^,()]+,Z\)|mean\([^,()]+\)~const)$"
)


@dataclass
class _Node:
    stmt: str
    parent: "_Node | None" = None
    children: list = field(default_factory=list)
    visits: int = 0
    sum_surprisal: float = 0.0
    eids: list[int] = field(default_factory=list)
    asserted: bool = False
    last_surprisal: float = 0.0


class AutoDiscoveryAgent:
    name = "AutoDisc-algo"

    def __init__(self, model: str | None = None, max_iterations: int = 12,
                 n_belief_samples: int = 3, k_pw: float = 1.0,
                 alpha_pw: float = 0.5, surprisal_kl: float = 0.1,
                 ucb_c: float = 1.0, llm_call_cap: int = 40):
        self.model = model or os.getenv("AUTODISC_MODEL", "openai/gpt-4o-mini")
        self.max_iter = max_iterations
        self.n = n_belief_samples
        self.k = k_pw
        self.alpha = alpha_pw
        self.kl_thresh = surprisal_kl
        self.ucb_c = ucb_c
        self.llm_cap = llm_call_cap
        self.calls = 0
        from openai import OpenAI
        self._c = OpenAI()

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def run(self, world: World, store: ClaimStore) -> None:
        self._w = world
        self._s = store
        subs = world._scm.structure.subdomains          # allowed surface
        self._view = {cid: list(vs) for cid, vs in subs.items()}
        root = _Node(stmt="")
        try:
            for _ in range(self.max_iter):
                if self._w.budget_left < 4.0 or self.calls >= self.llm_cap:
                    break
                sel = self._select(root)
                child = self._expand(sel)
                if child is None:
                    continue
                self._evaluate(child)
                self._backprop(child)
        except BudgetExhausted:
            pass

    # ------------------------------------------------------------------
    # MCTS
    # ------------------------------------------------------------------
    def _ucb(self, parent: _Node, child: _Node) -> float:
        if child.visits == 0:
            return float("inf")
        avg = child.sum_surprisal / child.visits
        explore = self.ucb_c * math.sqrt(
            2.0 * math.log(parent.visits + 1) / child.visits)
        return avg + explore

    def _select(self, root: _Node) -> _Node:
        node = root
        while node.children and \
                len(node.children) >= self.k * ((node.visits + 1) ** self.alpha):
            node = max(node.children, key=lambda c: self._ucb(node, c))
        return node

    def _expand(self, parent: _Node) -> _Node | None:
        path = []
        n = parent
        while n is not None and n.stmt:
            path.append(n.stmt); n = n.parent
        stmt = self._propose(path)
        if not stmt or not GRAMMAR_RE.match(stmt):
            return None
        if any(c.stmt == stmt for c in parent.children):
            return None
        child = _Node(stmt=stmt, parent=parent)
        parent.children.append(child)
        return child

    def _backprop(self, node: _Node) -> None:
        n = node.parent
        while n is not None:
            n.visits += 1
            n.sum_surprisal += node.last_surprisal
            n = n.parent
        node.visits += 1

    # ------------------------------------------------------------------
    # LLM calls (hypothesis generation + belief elicitation)
    # ------------------------------------------------------------------
    def _ask(self, user: str, max_tokens: int = 80, temperature: float = 1.0):
        if self.calls >= self.llm_cap:
            return None
        self.calls += 1
        try:
            r = self._c.chat.completions.create(
                model=self.model, temperature=temperature,
                max_tokens=max_tokens,
                messages=[{"role": "system", "content": SYS},
                          {"role": "user", "content": user}])
            return (r.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"[AutoDisc-LLM] {e}")
            return None

    def _propose(self, path: list[str]) -> str | None:
        txt = self._ask(
            f"clusters: {self._view}\n"
            f"path of prior hypotheses: {path}\n"
            f"Budget left: {round(self._w.budget_left,1)}. Propose ONE new "
            f"hypothesis in the strict grammar; do not repeat path. "
            f"Reply JSON: {{\"statement\": \"...\"}}", max_tokens=80)
        if not txt:
            return None
        i, j = txt.find("{"), txt.rfind("}")
        if i < 0 or j < 0:
            return None
        try:
            return str(json.loads(txt[i:j + 1]).get("statement", "")
                       ).replace(" ", "")
        except Exception:
            return None

    def _elicit_belief(self, stmt: str, evidence: str | None
                       ) -> tuple[float, float]:
        true_n = false_n = 0
        for _ in range(self.n):
            txt = self._ask(
                f"Hypothesis: {stmt}\n"
                + (f"Experimental evidence: {evidence}\n" if evidence else "")
                + "Is the hypothesis TRUE in the world being studied? "
                  "Reply JSON: {\"belief\": true|false}",
                max_tokens=15, temperature=0.8)
            if not txt:
                continue
            low = txt.lower()
            if '"belief":true' in low.replace(" ", "") or "true" in low and "false" not in low:
                true_n += 1
            elif "false" in low:
                false_n += 1
        return (1.0 + true_n, 1.0 + false_n)

    # ------------------------------------------------------------------
    # experiment execution (world calls only; honest provenance)
    # ------------------------------------------------------------------
    def _experiment(self, node: _Node) -> str | None:
        s, w = node.stmt, self._w
        try:
            if s.startswith("causal(") or s.startswith("no_effect("):
                body = s[s.index("(") + 1:s.index(")")]
                parts = body.split(",")
                A, B = parts[0], parts[1]
                if A not in w._scm.structure.vars or B not in w._scm.structure.vars:
                    return None
                hi = w.intervene(A, +2.0, [B], n=6); node.eids.append(hi.eid)
                lo = w.intervene(A, -2.0, [B], n=6); node.eids.append(lo.eid)
                d = hi.mean(B) - lo.mean(B)
                return f"do({A})->{B}: delta={round(d,2)}"
            if s.startswith("confounded("):
                body = s[s.index("(") + 1:s.index(")")]
                A, B, _ = body.split(",")
                if A not in w._scm.structure.vars or B not in w._scm.structure.vars:
                    return None
                o = w.observe([A, B], n=8); node.eids.append(o.eid)
                hi = w.intervene(A, +2.0, [B], n=6); node.eids.append(hi.eid)
                return (f"corr({A},{B})={round(o.corr(A,B),2)}, "
                        f"do({A})->{B}: delta={round(hi.mean(B)-o.mean(B),2)}")
            if s.startswith("mean(") and s.endswith(")~const"):
                V = s[len("mean("):-len(")~const")]
                if V not in w._scm.structure.vars:
                    return None
                o = w.observe([V], n=8); node.eids.append(o.eid)
                return f"mean({V})={round(o.mean(V),2)}"
        except BudgetExhausted:
            raise
        except Exception:
            return None
        return None

    def _kl(self, post: tuple[float, float], prior: tuple[float, float]
            ) -> float:
        a1, b1 = post; a2, b2 = prior
        try:
            t1 = _betaln(a2, b2) - _betaln(a1, b1)
            t2 = (a1 - a2) * _digamma(a1) + (b1 - b2) * _digamma(b1)
            t3 = (a2 - a1 + b2 - b1) * _digamma(a1 + b1)
            return max(0.0, t1 + t2 + t3)
        except Exception:
            return 0.0

    def _evaluate(self, node: _Node) -> None:
        prior = self._elicit_belief(node.stmt, evidence=None)
        evidence = self._experiment(node)
        if evidence is None:
            node.last_surprisal = 0.0
            return
        posterior = self._elicit_belief(node.stmt, evidence=evidence)
        kl = self._kl(posterior, prior)
        node.last_surprisal = kl
        if kl >= self.kl_thresh and not node.asserted and node.eids:
            conf = min(0.95, 0.6 + kl / 4.0)
            self._s.assert_claim(
                node.stmt, confidence=conf, provenance=node.eids[:4],
                budget=self._w.budget_total - self._w.budget_left)
            node.asserted = True
