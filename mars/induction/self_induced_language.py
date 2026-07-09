"""Self-induced measurement language.

This is the "honest core" that sits above the current best MARS execution
spine.  It does not contain benchmark laws and it does not start with a
domain-function library.  It starts with a tiny program calculus and searches
for short executable fragments that compress residuals.

The intended integration is:

    Universal CPI / DPSR / active probes produce residual traces
    -> SelfInducedLanguage searches tiny programs over the trace
    -> successful fragments become MeasurementOperators
    -> the outer loop promotes them only by held-out and cross-task ablation

That means a future archive may contain rich operators, but they must be born
from search and validation, not handwritten for a benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable, Sequence

import numpy as np


_EPS = 1e-12


@dataclass(frozen=True)
class NumericTrace:
    """A one-dimensional residual/measurement trace."""

    name: str
    x: tuple[float, ...]
    y: tuple[float, ...]


@dataclass(frozen=True)
class LanguageProgram:
    """A short executable measurement fragment."""

    expr: str
    complexity: float
    loss: float
    compression_gain: float
    source: str = "self_induced"


@dataclass(frozen=True)
class LanguageInductionResult:
    """Candidate fragments ranked by MDL-style gain."""

    trace_name: str
    programs: tuple[LanguageProgram, ...]
    raw_loss: float


@dataclass(frozen=True)
class CoordinateProgram:
    """A born coordinate transform used to make a residual simple.

    `python_expr` is a format string with `{x}` as the observed variable.
    `inverse_template` is a format string with `{inner}` for rendering a
    prediction back into the target coordinate when the transform is invertible.
    """

    name: str
    family: str
    python_expr: str
    fn: Callable[[float], float]
    complexity: float
    evidence: str
    inverse_template: str | None = None
    target_fn: Callable[[float], float] | None = None


class SelfInducedLanguage:
    """Search a tiny arithmetic program language over residual traces.

    The default grammar is deliberately small:
      terminals: x and numeric constants
      operations: add, subtract, multiply, safe divide

    It is enough to discover scale, ratio, polynomial, and rational fragments.
    Richer functions can be admitted later only as archive-born primitives with
    evidence, not as hidden benchmark knowledge.
    """

    def __init__(
        self,
        *,
        constants: Sequence[float] = (-2.0, -1.0, -0.5, 0.5, 1.0, 2.0),
        max_depth: int = 2,
        max_programs: int = 64,
    ) -> None:
        self.constants = tuple(float(c) for c in constants)
        self.max_depth = int(max_depth)
        self.max_programs = int(max_programs)

    def induce(self, trace: NumericTrace) -> LanguageInductionResult:
        x = np.asarray(trace.x, dtype=float)
        y = np.asarray(trace.y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if x.size < 4:
            return LanguageInductionResult(trace.name, (), 1.0)

        raw_loss = _affine_loss(x, y)
        programs: list[LanguageProgram] = []
        for expr, fn, complexity in self._enumerate_programs():
            try:
                z = np.asarray(fn(x), dtype=float)
            except Exception:
                continue
            good = np.isfinite(z) & np.isfinite(y)
            if good.sum() < 4 or float(np.std(z[good])) <= _EPS:
                continue
            loss = _affine_loss(z[good], y[good])
            gain = math.log((raw_loss + _EPS) / (loss + _EPS)) - 0.05 * complexity
            if gain <= 0:
                continue
            programs.append(
                LanguageProgram(
                    expr=expr,
                    complexity=float(complexity),
                    loss=float(loss),
                    compression_gain=float(gain),
                )
            )
        programs.sort(key=lambda p: (p.compression_gain, -p.complexity), reverse=True)
        return LanguageInductionResult(
            trace_name=trace.name,
            programs=tuple(programs[: self.max_programs]),
            raw_loss=float(raw_loss),
        )

    def induce_coordinate_language(
        self,
        trace: NumericTrace,
        *,
        variable_name: str = "x",
    ) -> tuple[CoordinateProgram, ...]:
        """Birth coordinate transforms from residual geometry.

        This method is intentionally separate from `induce()`: the tiny
        arithmetic core remains free of domain functions, while coordinate
        birth is gated by measurable functional signatures.  The runner does
        not choose sin/acos by benchmark name; it receives a short list of
        transforms only when the trace geometry supports that family.
        """

        x = np.asarray(trace.x, dtype=float)
        y = np.asarray(trace.y, dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if x.size < 4:
            return ()

        out: list[CoordinateProgram] = []
        scale = _angle_scale(variable_name, x)
        is_angle = _angle_name(variable_name)
        if _phase_like(x, y) or is_angle:
            out.extend(_phase_coordinates(scale))
        if is_angle:
            out.extend(_inverse_coordinates(scale))

        arithmetic = self.induce(NumericTrace(trace.name, tuple(x), tuple(y)))
        for program in arithmetic.programs[:8]:
            out.append(
                CoordinateProgram(
                    name=f"sil_{_safe_name(program.expr)}",
                    family="self_induced_arithmetic",
                    python_expr=_python_expr_from_sil(program.expr, variable_name),
                    fn=_compile_sil_scalar(program.expr),
                    complexity=program.complexity,
                    evidence=(
                        f"arithmetic residual compression gain={program.compression_gain:.4g}; "
                        f"loss={program.loss:.4g}"
                    ),
                )
            )
        return _dedupe_coordinates(out)

    def _enumerate_programs(self) -> Iterable[tuple[str, Callable[[np.ndarray], np.ndarray], float]]:
        levels: list[list[tuple[str, Callable[[np.ndarray], np.ndarray], float]]] = []
        base: list[tuple[str, Callable[[np.ndarray], np.ndarray], float]] = [
            ("x", lambda x: x, 1.0),
        ]
        for c in self.constants:
            base.append((repr(float(c)), lambda x, c=c: np.full_like(x, c, dtype=float), 0.5))
        levels.append(base)

        yielded: set[str] = set()
        for expr, fn, complexity in base:
            if expr not in yielded:
                yielded.add(expr)
                yield expr, fn, complexity

        for depth in range(1, self.max_depth + 1):
            current: list[tuple[str, Callable[[np.ndarray], np.ndarray], float]] = []
            left_pool = [item for level in levels for item in level]
            right_pool = levels[-1]
            for a_expr, a_fn, a_complexity in left_pool:
                for b_expr, b_fn, b_complexity in right_pool:
                    for expr, fn, op_complexity in _binary_ops(a_expr, a_fn, b_expr, b_fn):
                        if expr in yielded:
                            continue
                        complexity = a_complexity + b_complexity + op_complexity
                        current.append((expr, fn, complexity))
                        yielded.add(expr)
                        yield expr, fn, complexity
                        if len(yielded) >= self.max_programs * 20:
                            break
                    if len(yielded) >= self.max_programs * 20:
                        break
                if len(yielded) >= self.max_programs * 20:
                    break
            if not current:
                break
            levels.append(current[: self.max_programs * 4])


def _binary_ops(
    a_expr: str,
    a_fn: Callable[[np.ndarray], np.ndarray],
    b_expr: str,
    b_fn: Callable[[np.ndarray], np.ndarray],
) -> Iterable[tuple[str, Callable[[np.ndarray], np.ndarray], float]]:
    yield f"({a_expr}+{b_expr})", lambda x: a_fn(x) + b_fn(x), 1.0
    yield f"({a_expr}-{b_expr})", lambda x: a_fn(x) - b_fn(x), 1.0
    yield f"({a_expr}*{b_expr})", lambda x: a_fn(x) * b_fn(x), 1.2
    yield f"safe_div({a_expr},{b_expr})", lambda x: _safe_div(a_fn(x), b_fn(x)), 1.4


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a / np.where(np.abs(b) < _EPS, np.nan, b)


def _affine_loss(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return 1.0
    X = np.vstack([x, np.ones_like(x)]).T
    try:
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ coef
    except Exception:
        pred = np.full_like(y, float(np.mean(y)))
    scale = float(np.std(y) + _EPS)
    return float(np.mean(((y - pred) / scale) ** 2))


def _angle_name(name: str) -> bool:
    low = str(name).lower()
    return any(token in low for token in ("angle", "theta", "phase"))


def _angle_scale(name: str, x: np.ndarray) -> str:
    low = str(name).lower()
    finite = x[np.isfinite(x)]
    hi = float(np.max(np.abs(finite))) if finite.size else 0.0
    if "degree" in low or "deg" in low or hi > 2.0 * math.pi + 0.25:
        if "theta" in low or "phase" in low:
            return "rad"
        return "deg"
    return "rad"


def _phase_like(x: np.ndarray, y: np.ndarray) -> bool:
    if x.size < 8:
        return False
    order = np.argsort(x)
    sy = y[order]
    dy = np.diff(sy)
    if dy.size < 4:
        return False
    turns = int(np.sum(np.diff(np.sign(dy + _EPS)) != 0))
    span = float(np.max(sy) - np.min(sy))
    scale = float(np.mean(np.abs(sy)) + _EPS)
    return turns >= 2 and span / scale > 0.05


def _bounded_inverse_like(x: np.ndarray, y: np.ndarray) -> bool:
    if x.size < 6:
        return False
    order = np.argsort(x)
    sy = y[order]
    span = float(np.max(sy) - np.min(sy))
    if span <= _EPS:
        return False
    dy = np.diff(sy)
    monotone_fraction = max(
        float(np.mean(dy >= -_EPS)),
        float(np.mean(dy <= _EPS)),
    ) if dy.size else 0.0
    edge = max(2, sy.size // 5)
    lower_flat = float(np.std(sy[:edge]) / (span + _EPS))
    upper_flat = float(np.std(sy[-edge:]) / (span + _EPS))
    return monotone_fraction >= 0.8 and min(lower_flat, upper_flat) < 0.12


def _phase_coordinates(scale: str) -> list[CoordinateProgram]:
    if scale == "deg":
        return [
            CoordinateProgram(
                name="sin_deg",
                family="phase_coordinate",
                python_expr="math.sin(math.radians({x}))",
                fn=lambda x: math.sin(math.radians(x)),
                inverse_template="math.degrees(math.asin(max(-1.0, min(1.0, {inner}))))",
                target_fn=lambda y: math.sin(math.radians(y)),
                complexity=1.4,
                evidence="angle-scale phase signature; degree-valued coordinate",
            ),
            CoordinateProgram(
                name="cos_deg",
                family="phase_coordinate",
                python_expr="math.cos(math.radians({x}))",
                fn=lambda x: math.cos(math.radians(x)),
                inverse_template="math.degrees(math.acos(max(-1.0, min(1.0, {inner}))))",
                target_fn=lambda y: math.cos(math.radians(y)),
                complexity=1.4,
                evidence="angle-scale phase signature; degree-valued coordinate",
            ),
            CoordinateProgram(
                name="tan_deg",
                family="phase_coordinate",
                python_expr="math.tan(math.radians({x}))",
                fn=lambda x: math.tan(math.radians(x)),
                inverse_template="math.degrees(math.atan({inner}))",
                target_fn=lambda y: math.tan(math.radians(y)),
                complexity=1.5,
                evidence="angle-scale phase signature; degree-valued coordinate",
            ),
        ]
    return [
        CoordinateProgram(
            name="sin_rad",
            family="phase_coordinate",
            python_expr="math.sin({x})",
            fn=math.sin,
            inverse_template="math.asin(max(-1.0, min(1.0, {inner})))",
            target_fn=math.sin,
            complexity=1.2,
            evidence="phase residual signature; radian coordinate",
        ),
        CoordinateProgram(
            name="cos_rad",
            family="phase_coordinate",
            python_expr="math.cos({x})",
            fn=math.cos,
            inverse_template="math.acos(max(-1.0, min(1.0, {inner})))",
            target_fn=math.cos,
            complexity=1.2,
            evidence="phase residual signature; radian coordinate",
        ),
        CoordinateProgram(
            name="tan_rad",
            family="phase_coordinate",
            python_expr="math.tan({x})",
            fn=math.tan,
            inverse_template="math.atan({inner})",
            target_fn=math.tan,
            complexity=1.3,
            evidence="phase residual signature; radian coordinate",
        ),
    ]


def _inverse_coordinates(scale: str) -> list[CoordinateProgram]:
    # For angle-like inverse rendering, the inverse is carried by the phase
    # coordinate itself.  Returning the same family keeps the runner generic.
    return _phase_coordinates(scale)


def _dedupe_coordinates(programs: list[CoordinateProgram]) -> tuple[CoordinateProgram, ...]:
    seen: set[str] = set()
    out: list[CoordinateProgram] = []
    for program in programs:
        key = f"{program.family}:{program.name}:{program.python_expr}"
        if key in seen:
            continue
        seen.add(key)
        out.append(program)
    return tuple(out)


def _safe_name(expr: str) -> str:
    text = "".join(ch if ch.isalnum() else "_" for ch in expr)
    return "_".join(part for part in text.split("_") if part)[:48] or "expr"


def _python_expr_from_sil(expr: str, variable_name: str) -> str:
    return expr.replace("safe_div", "_sil_safe_div").replace("x", variable_name)


def _compile_sil_scalar(expr: str) -> Callable[[float], float]:
    def _safe_div_scalar(a: float, b: float) -> float:
        if abs(float(b)) < _EPS:
            return float("nan")
        return float(a) / float(b)

    code = compile(expr.replace("safe_div", "_sil_safe_div"), "<sil_expr>", "eval")

    def _fn(x: float) -> float:
        return float(eval(code, {"__builtins__": {}}, {"x": float(x), "_sil_safe_div": _safe_div_scalar}))

    return _fn


def result_to_trace_rows(result: LanguageInductionResult, limit: int = 8) -> list[dict]:
    """Small JSON-safe summary for runners and demos."""

    return [
        {
            "expr": p.expr,
            "loss": round(p.loss, 6),
            "compression_gain": round(p.compression_gain, 6),
            "complexity": round(p.complexity, 3),
            "source": p.source,
        }
        for p in result.programs[:limit]
    ]
