"""Trainable local hypothesis prior for MARS-NOVA."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Sequence

from mars.nova.oracle_tasks import OracleTask


@dataclass(frozen=True)
class AnalyzerSketch:
    """A candidate hypothesis constructor before sandbox compilation."""

    name: str
    description: str
    code: str
    predict: Callable[[Any, dict[str, Any]], Any]
    complexity: float = 2.0
    tags: tuple[str, ...] = ("nova",)


@dataclass(frozen=True)
class PriorFitResult:
    """Posterior ranking after local oracle supervision."""

    ranked: tuple[tuple[AnalyzerSketch, float, float], ...]
    trace: tuple[dict[str, Any], ...]


class HypothesisPrior:
    """Bayesian-style trainable prior over executable analyzer sketches.

    v0 is deliberately lightweight: it updates posterior weights from oracle
    losses rather than changing neural weights.  This still moves hypothesis
    selection out of free-form LLM guessing and into measured local supervision.
    """

    def __init__(
        self,
        *,
        temperature: float = 0.25,
        complexity_weight: float = 0.03,
    ):
        self.temperature = max(1e-6, float(temperature))
        self.complexity_weight = float(complexity_weight)

    def fit(
        self,
        sketches: Sequence[AnalyzerSketch],
        tasks: Sequence[OracleTask],
    ) -> PriorFitResult:
        rows: list[tuple[AnalyzerSketch, float, float]] = []
        trace: list[dict[str, Any]] = []
        for sketch in sketches:
            losses = []
            exact = 0
            for task in tasks:
                try:
                    pred = sketch.predict(task.inputs, dict(task.context))
                    loss = generic_loss(pred, task.target)
                except Exception:
                    loss = 1.0
                loss = min(1.0, max(0.0, loss))
                losses.append(loss)
                exact += int(loss <= 1e-9)
            mean_loss = sum(losses) / len(losses) if losses else 1.0
            objective = mean_loss + self.complexity_weight * sketch.complexity
            posterior = math.exp(-objective / self.temperature)
            rows.append((sketch, mean_loss, posterior))
            trace.append(
                {
                    "name": sketch.name,
                    "loss_mean": mean_loss,
                    "exact_rate": exact / len(losses) if losses else 0.0,
                    "complexity": sketch.complexity,
                    "posterior_unnormalized": posterior,
                }
            )
        total = sum(row[2] for row in rows) or 1.0
        ranked = tuple(
            sorted(
                ((sketch, loss, weight / total) for sketch, loss, weight in rows),
                key=lambda row: (row[1], -row[2], row[0].complexity, row[0].name),
            )
        )
        trace.sort(key=lambda row: (float(row["loss_mean"]), float(row["complexity"])))
        return PriorFitResult(ranked=ranked, trace=tuple(trace))


def generic_loss(prediction: Any, target: Any) -> float:
    """A benchmark-agnostic local oracle loss."""

    if isinstance(target, (int, float)) and not isinstance(target, bool):
        try:
            p = float(prediction)
            t = float(target)
        except Exception:
            return 1.0
        denom = abs(t) + 1e-9
        return min(1.0, abs(p - t) / denom)
    if isinstance(target, str):
        return 0.0 if str(prediction) == target else _string_loss(str(prediction), target)
    if isinstance(target, dict):
        if not isinstance(prediction, dict):
            return 1.0
        keys = set(target) | set(prediction)
        if not keys:
            return 0.0
        misses = sum(1 for key in keys if prediction.get(key) != target.get(key))
        return min(1.0, misses / len(keys))
    return 0.0 if prediction == target else 1.0


def _string_loss(prediction: str, target: str) -> float:
    if prediction == target:
        return 0.0
    if not prediction and not target:
        return 0.0
    denom = max(1, max(len(prediction), len(target)))
    prefix = 0
    for a, b in zip(prediction, target):
        if a != b:
            break
        prefix += 1
    length_penalty = abs(len(prediction) - len(target)) / denom
    prefix_penalty = 1.0 - prefix / denom
    return min(1.0, 0.7 * prefix_penalty + 0.3 * length_penalty)
