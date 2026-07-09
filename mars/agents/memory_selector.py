"""
MemorySelector — choose top-k claims to show the Generator.

This is the central design fix for the OLS v0.2 failure mode: instead of
dumping the entire ClaimStore into Generator's context (which we proved
hurt gpt-4o-mini by +0.24 RPS), we surface only the most relevant K claims
per sub-goal.

v0.1 implementation: pure heuristic — no LLM call, no embedding API.
  score(claim, sub_goal) = w_recency · recency + w_overlap · token_overlap
                           + w_confidence · confidence

  - recency = exp(−(B − claim.budget_stamp) / B) ∈ (0, 1]  (newer = higher)
  - token_overlap = |tokens(claim.statement) ∩ tokens(sub_goal.question)|
                    / max(1, |tokens(sub_goal.question)|)
  - confidence = claim.confidence ∈ [0, 1]

Defaults: w_recency=0.4, w_overlap=0.4, w_confidence=0.2.

v0.2 plan: implement CMI-style causal-intervention selection (estimate how
each candidate memory changes Generator's predicted output on a probe
question, drop those that don't help or harm). See arxiv 2605.17641.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from ols.core.types import Claim


_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-я0-9_.+-]+")


def _tokens(s: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(s or "") if len(t) > 1}


@dataclass
class MemorySelector:
    k: int = 5
    w_recency: float = 0.4
    w_overlap: float = 0.4
    w_confidence: float = 0.2

    def select(
        self,
        claims: list[Claim],
        sub_goal_question: str,
        budget_total: float,
        current_sub_goal: str | None = None,
    ) -> list[Claim]:
        active = [c for c in claims if c.status == "active"]
        if not active:
            return []
        q_tokens = _tokens(sub_goal_question)
        scored: list[tuple[float, Claim]] = []
        B = max(1.0, budget_total)
        for c in active:
            recency = math.exp(-max(0.0, (B - c.budget_stamp)) / B)
            c_tokens = _tokens(c.statement)
            overlap = (len(q_tokens & c_tokens) / max(1, len(q_tokens))
                       if q_tokens else 0.0)
            same_sg_bonus = 0.05 if (current_sub_goal and c.sub_goal == current_sg_or_none(current_sub_goal)) else 0.0
            score = (self.w_recency * recency
                     + self.w_overlap * overlap
                     + self.w_confidence * c.confidence
                     + same_sg_bonus)
            scored.append((score, c))
        scored.sort(key=lambda x: -x[0])
        return [c for _, c in scored[: self.k]]

    def render_for_log(self, picked: list[Claim]) -> str:
        if not picked:
            return "(empty)"
        return "; ".join(f"[{c.confidence:.2f}] {c.statement[:60]}"
                         for c in picked)


def current_sg_or_none(sg: str | None) -> str | None:
    return sg
