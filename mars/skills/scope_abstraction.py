"""Scope abstraction for measured hypotheses.

Some scientific questions name a local object but expect the answer to place
that object inside a broader invariant.  This layer decides when a local
measurement should be rendered alone and when it should be embedded in a
stable cross-group pattern.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Sequence


@dataclass(frozen=True)
class MeasuredPairGroup:
    label: str
    left_value: float
    right_value: float
    n: int

    @property
    def delta(self) -> float:
        return self.left_value - self.right_value


@dataclass(frozen=True)
class ScopeAbstraction:
    mode: str
    groups: tuple[str, ...]
    score: float
    evidence: str


def induce_scope_abstraction(
    *,
    question: str,
    requested_groups: Sequence[str],
    rows: Sequence[MeasuredPairGroup],
) -> ScopeAbstraction | None:
    """Infer whether a local answer should include a broader invariant."""

    clean = [r for r in rows if r.n > 0 and math.isfinite(r.left_value) and math.isfinite(r.right_value)]
    if len(clean) < 2 or not requested_groups:
        return None
    q = question.lower()
    if any(token in q for token in ("only", "solely", "just this", "within this")):
        return None
    if not any(token in q for token in ("as compared", "compared to that", "larger", "smaller", "than", "versus", " vs ")):
        return None
    requested = {_norm(g) for g in requested_groups}
    requested_rows = [r for r in clean if _norm(r.label) in requested]
    if not requested_rows:
        return None
    signs = [1 if r.delta > 0 else -1 if r.delta < 0 else 0 for r in clean]
    requested_signs = [1 if r.delta > 0 else -1 if r.delta < 0 else 0 for r in requested_rows]
    if not requested_signs or requested_signs[0] == 0:
        return None
    same = [r for r, sign in zip(clean, signs) if sign == requested_signs[0]]
    support = len(same) / len(clean)
    if len(same) < 2 or support < 0.6:
        return None
    mean_abs_delta = sum(abs(r.delta) for r in same) / len(same)
    local_strength = sum(abs(r.delta) for r in requested_rows) / max(1, len(requested_rows))
    score = 2.0 + 2.5 * support + min(2.0, 4.0 * mean_abs_delta) + min(1.5, 4.0 * local_strength)
    evidence = (
        f"scope_local_plus_global:support={support:.4g}:"
        f"requested={','.join(r.label for r in requested_rows)}:"
        f"same_direction={','.join(r.label for r in same)}:"
        f"score={score:.4g}"
    )
    return ScopeAbstraction("local_plus_global", tuple(r.label for r in same), score, evidence)


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()
