"""Bayesian operator genome for self-building measurement languages.

This module is the bridge between the old MARS idea ("hypotheses are small
executable analyzers") and the missing mechanism ("where do good analyzers
come from when the inner model is weak?").

The core move is benchmark-agnostic:

    failed measurements -> residual geometry -> new measurement operator

The archive stores operators, not benchmark answers.  An operator is a small
measurement family with:
  - an activation signature: when residuals suggest it is useful;
  - an evidence model: held-out loss, invariance, compression, stability;
  - a transfer score: whether it repeatedly helps across tasks.

The math is intentionally lightweight here.  We use an MDL/Bayes-factor style
score that can be computed from ordinary traces without knowing ground truth:

    posterior_score = compression_gain + invariance_gain
                      + stability_gain - complexity_penalty

Higher score means the operator made the world simpler after execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


_EPS = 1e-12


@dataclass(frozen=True)
class MeasurementTrace:
    """One small executed measurement and its residual pattern."""

    name: str
    xs: tuple[float, ...]
    ys: tuple[float, ...]
    prediction: tuple[float, ...] = ()
    context: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FailureGeometry:
    """Typed geometry extracted from a failed measurement."""

    kind: str
    severity: float
    evidence: str
    bits: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MeasurementOperator:
    """A reusable way of making a hidden hypothesis visible."""

    operator_id: str
    family: str
    activation_signature: str
    transform_name: str
    transform_expr: str
    expected_invariance: str
    complexity: float
    posterior_score: float
    evidence: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class OperatorGenomeArchive:
    """Simple persistent-in-memory operator archive.

    A later outer loop can serialize this, mutate it, and promote operators
    only when parent-relative ablations show positive official score movement.
    """

    operators: dict[str, MeasurementOperator] = field(default_factory=dict)
    wins: dict[str, float] = field(default_factory=dict)
    trials: dict[str, float] = field(default_factory=dict)

    def add(self, op: MeasurementOperator) -> None:
        old = self.operators.get(op.operator_id)
        if old is None or op.posterior_score > old.posterior_score:
            self.operators[op.operator_id] = op
        self.trials[op.operator_id] = self.trials.get(op.operator_id, 0.0) + 1.0
        if op.posterior_score > 0:
            self.wins[op.operator_id] = self.wins.get(op.operator_id, 0.0) + 1.0

    def ranked(self, limit: int = 16) -> list[MeasurementOperator]:
        def score(op: MeasurementOperator) -> float:
            trials = self.trials.get(op.operator_id, 1.0)
            win_rate = self.wins.get(op.operator_id, 0.0) / max(trials, 1.0)
            return op.posterior_score + math.log1p(trials) * win_rate

        return sorted(self.operators.values(), key=score, reverse=True)[:limit]


def infer_failure_geometry(trace: MeasurementTrace) -> tuple[FailureGeometry, ...]:
    """Convert a measurement trace into domain-agnostic residual syndromes."""

    xs = np.asarray(trace.xs, dtype=float)
    ys = np.asarray(trace.ys, dtype=float)
    if xs.size < 4 or ys.size != xs.size:
        return ()
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[mask], ys[mask]
    if xs.size < 4:
        return ()

    out: list[FailureGeometry] = []
    order = np.argsort(xs)
    sx, sy = xs[order], ys[order]
    scale = float(np.mean(np.abs(sy)) + _EPS)
    rough = float(np.std(np.diff(sy)) / scale) if sy.size > 2 else 0.0

    sign_changes = int(np.sum(np.diff(np.sign(np.diff(sy) + _EPS)) != 0)) if sy.size > 3 else 0
    if sign_changes >= 2:
        out.append(
            FailureGeometry(
                kind="oscillatory_or_phase_residual",
                severity=min(1.0, sign_changes / max(3.0, sy.size / 3.0)),
                evidence="ordered measurements change curvature/sign repeatedly",
                bits={"sign_changes": sign_changes, "roughness": rough},
            )
        )

    y_min, y_max = float(np.min(sy)), float(np.max(sy))
    span = y_max - y_min
    if span > _EPS:
        lower = float(np.mean(np.abs(sy[: max(2, sy.size // 5)] - y_min)) / span)
        upper = float(np.mean(np.abs(sy[-max(2, sy.size // 5):] - y_max)) / span)
        if min(lower, upper) < 0.12 and rough < 0.8:
            out.append(
                FailureGeometry(
                    kind="saturating_or_inverse_coordinate",
                    severity=1.0 - min(lower, upper),
                    evidence="measurements approach a boundary after monotone change",
                    bits={"lower_boundary_error": lower, "upper_boundary_error": upper},
                )
            )

    abs_y = np.abs(sy)
    finite_positive = abs_y[abs_y > _EPS]
    if finite_positive.size >= 4 and float(np.max(finite_positive) / np.min(finite_positive)) > 1e4:
        out.append(
            FailureGeometry(
                kind="singular_or_exponential_scale",
                severity=1.0,
                evidence="target spans several orders of magnitude",
                bits={"dynamic_range": float(np.max(finite_positive) / np.min(finite_positive))},
            )
        )

    if _has_step_like_split(sx, sy):
        out.append(
            FailureGeometry(
                kind="regime_split",
                severity=0.85,
                evidence="one threshold explains a large mean shift",
                bits={},
            )
        )

    return tuple(out)


def induce_univariate_operators(trace: MeasurementTrace) -> tuple[MeasurementOperator, ...]:
    """Birth measurement operators from a failed univariate trace.

    This does not know NewtonBench, DiscoveryBench, or UltraHorizon.  It only
    sees a small vector of measurements and asks which coordinate system makes
    the relation simpler.
    """

    xs = np.asarray(trace.xs, dtype=float)
    ys = np.asarray(trace.ys, dtype=float)
    mask = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[mask], ys[mask]
    if xs.size < 4:
        return ()

    geometries = infer_failure_geometry(
        MeasurementTrace(name=trace.name, xs=tuple(xs), ys=tuple(ys), context=trace.context)
    )
    candidates = _candidate_charts(geometries)
    if not candidates:
        candidates = _candidate_charts(
            (FailureGeometry("generic_coordinate_search", 0.3, "fallback chart search"),)
        )

    raw_loss = _best_affine_loss(xs, ys)
    ops: list[MeasurementOperator] = []
    for family, name, expr, fn, complexity in candidates:
        try:
            tx = np.asarray(fn(xs), dtype=float)
        except Exception:
            continue
        good = np.isfinite(tx) & np.isfinite(ys)
        if good.sum() < 4 or float(np.std(tx[good])) <= _EPS:
            continue
        chart_loss = _best_affine_loss(tx[good], ys[good])
        log_chart_loss = _best_log_affine_loss(tx[good], ys[good])
        best_loss = min(chart_loss, log_chart_loss)
        compression_gain = math.log((raw_loss + _EPS) / (best_loss + _EPS))
        stability_gain = _bootstrap_stability(tx[good], ys[good])
        invariance_gain = _monotone_invariance_gain(tx[good], ys[good])
        posterior = compression_gain + 0.25 * stability_gain + 0.15 * invariance_gain - 0.20 * complexity
        if posterior <= 0.05:
            continue
        geom = max(geometries, key=lambda g: g.severity, default=None)
        ops.append(
            MeasurementOperator(
                operator_id=f"{family}:{name}",
                family=family,
                activation_signature=geom.kind if geom else "generic_coordinate_search",
                transform_name=name,
                transform_expr=expr,
                expected_invariance="relation becomes lower-loss and stable after coordinate transform",
                complexity=float(complexity),
                posterior_score=float(posterior),
                evidence={
                    "raw_loss": float(raw_loss),
                    "chart_loss": float(best_loss),
                    "compression_gain": float(compression_gain),
                    "stability_gain": float(stability_gain),
                    "invariance_gain": float(invariance_gain),
                },
            )
        )
    return tuple(sorted(ops, key=lambda o: o.posterior_score, reverse=True))


def _candidate_charts(
    geometries: Sequence[FailureGeometry],
) -> list[tuple[str, str, str, Callable[[np.ndarray], np.ndarray], float]]:
    kinds = {g.kind for g in geometries}
    charts: list[tuple[str, str, str, Callable[[np.ndarray], np.ndarray], float]] = []
    if "oscillatory_or_phase_residual" in kinds:
        charts.extend(
            [
                ("phase_chart", "sin(x)", "sin(x)", np.sin, 1.2),
                ("phase_chart", "cos(x)", "cos(x)", np.cos, 1.2),
                ("phase_chart", "sin(2x)", "sin(2*x)", lambda x: np.sin(2 * x), 1.4),
                ("phase_chart", "cos(2x)", "cos(2*x)", lambda x: np.cos(2 * x), 1.4),
            ]
        )
    if "saturating_or_inverse_coordinate" in kinds:
        charts.extend(
            [
                ("inverse_coordinate_chart", "asin_clip(x)", "asin(clip(x,-1,1))", lambda x: np.arcsin(np.clip(x, -1, 1)), 1.6),
                ("inverse_coordinate_chart", "acos_clip(x)", "acos(clip(x,-1,1))", lambda x: np.arccos(np.clip(x, -1, 1)), 1.6),
                ("inverse_coordinate_chart", "logit_unit(x)", "log(x/(1-x))", _logit_unit, 1.7),
            ]
        )
    if "singular_or_exponential_scale" in kinds:
        charts.extend(
            [
                ("singular_chart", "log1p(x)", "log1p(x)", np.log1p, 1.1),
                ("singular_chart", "1/x", "1/x", lambda x: 1.0 / np.maximum(np.abs(x), 1e-9), 1.2),
                ("singular_chart", "expm1(x)", "expm1(x)", lambda x: np.expm1(np.clip(x, -50, 50)), 1.5),
                ("singular_chart", "bose_denominator", "1/(expm1(x))", lambda x: 1.0 / np.maximum(np.expm1(np.clip(x, -50, 50)), 1e-9), 2.0),
            ]
        )
    if "regime_split" in kinds:
        charts.extend(
            [
                ("regime_chart", "threshold_indicator", "I[x >= tau]", _best_threshold_indicator, 1.4),
            ]
        )
    if not charts:
        charts.extend(
            [
                ("scale_chart", "x", "x", lambda x: x, 0.5),
                ("scale_chart", "log(x)", "log(x)", lambda x: np.log(np.maximum(np.abs(x), 1e-9)), 0.8),
                ("scale_chart", "sqrt(x)", "sqrt(x)", lambda x: np.sqrt(np.maximum(x, 0)), 0.9),
                ("scale_chart", "x^2", "x**2", lambda x: x * x, 0.9),
            ]
        )
    return charts


def _best_affine_loss(xs: np.ndarray, ys: np.ndarray) -> float:
    if xs.size < 2:
        return 1.0
    X = np.vstack([xs, np.ones_like(xs)]).T
    try:
        coef, *_ = np.linalg.lstsq(X, ys, rcond=None)
        pred = X @ coef
    except Exception:
        pred = np.full_like(ys, float(np.mean(ys)))
    scale = float(np.std(ys) + _EPS)
    return float(np.mean(((ys - pred) / scale) ** 2))


def _best_log_affine_loss(xs: np.ndarray, ys: np.ndarray) -> float:
    good = (np.abs(xs) > _EPS) & (np.abs(ys) > _EPS)
    if good.sum() < 4:
        return 1.0
    lx = np.log(np.abs(xs[good]))
    ly = np.log(np.abs(ys[good]))
    return _best_affine_loss(lx, ly)


def _bootstrap_stability(xs: np.ndarray, ys: np.ndarray, n: int = 12) -> float:
    if xs.size < 8:
        return 0.0
    rng = np.random.default_rng(123)
    coefs = []
    for _ in range(n):
        idx = rng.integers(0, xs.size, size=xs.size)
        X = np.vstack([xs[idx], np.ones_like(xs[idx])]).T
        try:
            coef, *_ = np.linalg.lstsq(X, ys[idx], rcond=None)
            coefs.append(float(coef[0]))
        except Exception:
            pass
    if len(coefs) < 3:
        return 0.0
    denom = abs(float(np.mean(coefs))) + _EPS
    cv = float(np.std(coefs) / denom)
    return float(max(0.0, 1.0 - min(1.0, cv)))


def _monotone_invariance_gain(xs: np.ndarray, ys: np.ndarray) -> float:
    if xs.size < 4:
        return 0.0
    rx = np.argsort(np.argsort(xs))
    ry = np.argsort(np.argsort(ys))
    corr = np.corrcoef(rx, ry)[0, 1]
    if not np.isfinite(corr):
        return 0.0
    return float(abs(corr))


def _has_step_like_split(xs: np.ndarray, ys: np.ndarray) -> bool:
    if xs.size < 8:
        return False
    order = np.argsort(xs)
    sy = ys[order]
    total_var = float(np.var(sy) + _EPS)
    best = 0.0
    for i in range(2, sy.size - 2):
        left, right = sy[:i], sy[i:]
        within = (len(left) * np.var(left) + len(right) * np.var(right)) / len(sy)
        best = max(best, 1.0 - float(within / total_var))
    return best > 0.75


def _best_threshold_indicator(xs: np.ndarray) -> np.ndarray:
    if xs.size == 0:
        return xs
    tau = float(np.median(xs))
    return (xs >= tau).astype(float)


def _logit_unit(xs: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(xs)), float(np.max(xs))
    span = max(hi - lo, _EPS)
    u = np.clip((xs - lo) / span, 1e-6, 1 - 1e-6)
    return np.log(u / (1 - u))


def summarize_operator_genome(ops: Iterable[MeasurementOperator], limit: int = 5) -> list[dict[str, Any]]:
    """Small JSON-safe trace for benchmark runners and demos."""

    out = []
    for op in list(ops)[:limit]:
        out.append(
            {
                "operator_id": op.operator_id,
                "family": op.family,
                "activation_signature": op.activation_signature,
                "transform": op.transform_expr,
                "posterior_score": round(op.posterior_score, 4),
                "evidence": {
                    k: round(float(v), 4) if isinstance(v, (int, float)) else v
                    for k, v in op.evidence.items()
                },
            }
        )
    return out
