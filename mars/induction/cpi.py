"""Core objects for self-building causal/program induction.

The point of this layer is to make a hypothesis measurable. A hypothesis is a
small executable analyzer/program, not an opaque sentence. It predicts one
observable transition, is scored against counterexamples, and competes under a
simple MDL-style objective.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence


ProgramFn = Callable[[Any, dict[str, Any]], Any]


@dataclass(frozen=True)
class RuleTrace:
    """One transition to be explained by a hypothesis program."""

    current: Any
    target: Any
    context: dict[str, Any]


@dataclass
class ProgramHypothesis:
    """Executable hypothesis with lightweight causal/program metadata."""

    name: str
    description: str
    fn: ProgramFn
    complexity: float = 1.0
    causal_anchors: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    evidence: list[dict[str, Any]] = field(default_factory=list)

    def predict(self, trace: RuleTrace) -> Any:
        return self.fn(trace.current, trace.context)


@dataclass(frozen=True)
class HypothesisScore:
    name: str
    loss_mean: float
    exact_rate: float
    mdl_score: float
    n: int
    complexity: float


def _as_stable_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def edit_distance(a: str, b: str) -> int:
    """Small dependency-free Levenshtein distance."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(
                min(
                    prev[j] + 1,
                    cur[j - 1] + 1,
                    prev[j - 1] + (0 if ca == cb else 1),
                )
            )
        prev = cur
    return prev[-1]


def prediction_loss(prediction: Any, target: Any) -> float:
    """Normalized exact/edit loss for generic observable predictions."""
    if prediction == target:
        return 0.0
    p = _as_stable_text(prediction)
    t = _as_stable_text(target)
    denom = max(1, len(p), len(t))
    return edit_distance(p, t) / denom


def score_hypothesis(
    hypothesis: ProgramHypothesis,
    traces: Sequence[RuleTrace],
    *,
    complexity_weight: float = 0.01,
) -> HypothesisScore:
    losses: list[float] = []
    exact = 0
    for trace in traces:
        try:
            pred = hypothesis.predict(trace)
            loss = prediction_loss(pred, trace.target)
        except Exception:
            pred = None
            loss = 1.0
        exact += int(loss == 0.0)
        losses.append(loss)
    mean = sum(losses) / len(losses) if losses else 1.0
    return HypothesisScore(
        name=hypothesis.name,
        loss_mean=mean,
        exact_rate=exact / len(losses) if losses else 0.0,
        mdl_score=mean + complexity_weight * hypothesis.complexity,
        n=len(losses),
        complexity=hypothesis.complexity,
    )


def rank_hypotheses(
    hypotheses: Iterable[ProgramHypothesis],
    traces: Sequence[RuleTrace],
    *,
    complexity_weight: float = 0.01,
) -> list[tuple[ProgramHypothesis, HypothesisScore]]:
    ranked = [
        (
            h,
            score_hypothesis(
                h,
                traces,
                complexity_weight=complexity_weight,
            ),
        )
        for h in hypotheses
    ]
    ranked.sort(key=lambda x: (x[1].mdl_score, x[1].loss_mean, x[0].complexity))
    return ranked


def disagreement_score(predictions: Sequence[Any]) -> float:
    """Expected refutation value proxy for selecting the next experiment."""
    if len(predictions) < 2:
        return 0.0
    losses: list[float] = []
    for i, left in enumerate(predictions):
        for right in predictions[i + 1 :]:
            losses.append(prediction_loss(left, right))
    unique_bonus = len({_as_stable_text(p) for p in predictions}) - 1
    return (sum(losses) / len(losses)) + 0.05 * unique_bonus if losses else 0.0
