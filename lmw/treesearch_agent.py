"""
Tree-Search-LLM-Judge agent (algorithm-kernel adapter, ≈ AI Scientist v2 outer
loop).

Yamada et al. (arXiv:2504.08066) propose agentic tree search at workshop-level
ML research: a tree of research states, each expanded by an LLM proposing a
refinement, evaluated by an LLM-as-judge against quality criteria, with
best-first selection over the tree. The novel contribution is the ML-research
scaffolding (experiment_designer / code_writer / paper_writer prompts), not the
search algorithm; the search KERNEL — *tree search + LLM-as-judge* — is what
transfers cleanly to LMW's `observe/intervene` action space.

This is *not* a wrap of the original codebase. Label on the leaderboard:
"Tree-Search-LLM-Judge (adapter, ≈ AI Scientist v2 outer-loop kernel; not
original system)". See AUTODISCOVERY_ADAPTER_v0.md for the same labelling
discipline applied to AutoDiscovery.

Distinct from `autodisc_agent.py`:
  AutoDisc-algo uses MCTS with progressive widening, UCB exploration, and a
  Bayesian-surprise (KL-of-Beta) reward elicited from LLM belief sampling.
  Tree-Search-LLM-Judge uses best-first selection (no UCB explore term), no
  progressive widening, and a single LLM-judge scalar reward in [0,1].
  Same allowed surface, same claim grammar, same provenance discipline.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field

from claims import ClaimStore
from world import BudgetExhausted, World

SYS = (
    "You are an autonomous research agent in a small synthetic world with "
    "opaque variable clusters. You can observe (association only) and "
    "intervene (do(.) to reveal causation). Score by establishing TRUE "
    "high-value causal / null / confounded facts; AVOID confidently false "
    "claims (integrity penalty). Some clusters look promising but are inert "
    "— abandon them. Strict claim grammar (the ONLY allowed statement "
    "forms): causal(A,B,+) | causal(A,B,-) | no_effect(A,B) | "
    "confounded(A,B,Z) | mean(V)~const. Reply with EXACTLY one JSON object."
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
    expanded: bool = False
    judge: float = 0.0           # LLM-as-judge quality in [0, 1]
    evidence: str = ""
    eids: list[int] = field(default_factory=list)
    asserted: bool = False


class TreeSearchAgent:
    name = "TreeSearch-LLM-Judge"

    def __init__(self, model: str | None = None, max_iterations: int = 12,
                 assert_threshold: float = 0.7, branch_per_node: int = 2,
                 llm_call_cap: int = 40):
        self.model = model or os.getenv("TS_MODEL", "openai/gpt-4o-mini")
        self.max_iter = max_iterations
        self.assert_thresh = assert_threshold
        self.branch = branch_per_node
        self.llm_cap = llm_call_cap
        self.calls = 0
        from openai import OpenAI
        self._c = OpenAI()

    # ------------------------------------------------------------------
    def run(self, world: World, store: ClaimStore) -> None:
        self._w = world
        self._s = store
        subs = world._scm.structure.subdomains              # allowed surface
        self._view = {cid: list(vs) for cid, vs in subs.items()}
        root = _Node(stmt="", judge=0.5)
        frontier: list[_Node] = [root]
        try:
            for _ in range(self.max_iter):
                if self._w.budget_left < 4.0 or self.calls >= self.llm_cap:
                    break
                node = self._select(frontier)
                if node is None:
                    break
                for _b in range(self.branch):
                    if self.calls >= self.llm_cap or self._w.budget_left < 4.0:
                        break
                    child = self._expand(node)
                    if child is None:
                        continue
                    self._evaluate(child)
                    frontier.append(child)
                node.expanded = True
        except BudgetExhausted:
            pass

    # ------------------------------------------------------------------
    def _select(self, frontier: list[_Node]) -> _Node | None:
        # best-first: highest-judge unexpanded node
        cand = [n for n in frontier if not n.expanded]
        if not cand:
            return None
        return max(cand, key=lambda n: n.judge)

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

    # ------------------------------------------------------------------
    def _ask(self, user: str, max_tokens: int = 80, temperature: float = 0.7):
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
            print(f"[TreeSearch-LLM] {e}")
            return None

    def _propose(self, path: list[str]) -> str | None:
        txt = self._ask(
            f"clusters: {self._view}\nprior hypotheses on path: {path}\n"
            f"Budget left: {round(self._w.budget_left,1)}. Propose ONE new "
            f"hypothesis in strict grammar. Reply JSON: "
            f"{{\"statement\":\"...\"}}", max_tokens=80, temperature=1.0)
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

    def _experiment(self, node: _Node) -> str | None:
        s, w = node.stmt, self._w
        try:
            if s.startswith("causal(") or s.startswith("no_effect("):
                body = s[s.index("(") + 1:s.index(")")]
                A, B = body.split(",")[:2]
                if A not in w._scm.structure.vars or B not in w._scm.structure.vars:
                    return None
                hi = w.intervene(A, +2.0, [B], n=6); node.eids.append(hi.eid)
                lo = w.intervene(A, -2.0, [B], n=6); node.eids.append(lo.eid)
                return f"do({A})->{B}: delta={round(hi.mean(B)-lo.mean(B),2)}"
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

    def _judge(self, stmt: str, evidence: str | None) -> float:
        """LLM-as-judge: rate hypothesis quality given evidence in [0,1]."""
        txt = self._ask(
            f"Hypothesis: {stmt}\n"
            + (f"Experimental evidence: {evidence}\n" if evidence
               else "(no experimental evidence yet)\n")
            + "Rate the hypothesis's TRUTH and INFORMATIVENESS jointly on "
              "[0,1] (0=wrong/trivial, 1=clearly true and high-value). "
              "Reply JSON: {\"score\": <float>}",
            max_tokens=30, temperature=0.4)
        if not txt:
            return 0.0
        try:
            i, j = txt.find("{"), txt.rfind("}")
            v = float(json.loads(txt[i:j + 1]).get("score", 0.0))
            return max(0.0, min(1.0, v))
        except Exception:
            return 0.0

    def _evaluate(self, node: _Node) -> None:
        evidence = self._experiment(node)
        if evidence is None:
            node.judge = 0.0
            return
        node.evidence = evidence
        node.judge = self._judge(node.stmt, evidence)
        if node.judge >= self.assert_thresh and not node.asserted and node.eids:
            self._s.assert_claim(
                node.stmt, confidence=node.judge, provenance=node.eids[:4],
                budget=self._w.budget_total - self._w.budget_left)
            node.asserted = True
