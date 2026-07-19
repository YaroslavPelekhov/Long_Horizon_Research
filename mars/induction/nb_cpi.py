"""NewtonBench CPI: build analyzers that turn trajectories into law data.

This is the NewtonBench counterpart to the UltraHorizon Seq CPI prototype.
The hypothesis is a small measurement program:

    raw observation -> recovered physical quantity -> symbolic law candidate

For trajectory systems, the raw benchmark does not return the hidden law value
directly. CPI constructs an analyzer that recovers the force-like target from
finite differences, then fits compact programs under a refutation/MDL objective.
"""

from __future__ import annotations

import importlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mars.induction.cpi import ProgramHypothesis, RuleTrace, rank_hypotheses

_PROJ = Path(__file__).resolve().parent.parent.parent
_NB_REPO = _PROJ / "newtonbench_repo"
if str(_NB_REPO) not in sys.path:
    sys.path.insert(0, str(_NB_REPO))


@dataclass(frozen=True)
class NBForcePoint:
    inputs: dict[str, float]
    force: float
    raw_output: Any
    analyzer: str


@dataclass(frozen=True)
class LawCandidate:
    kind: str
    form: str
    python_body: str
    train_loss: float
    holdout_loss: float
    mdl_score: float
    complexity: float

    def function_code(self, signature: str) -> str:
        return f"{signature}\n    {self.python_body}"


def _float_array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _position_array(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def _velocity_array(values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def _safe_log_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (np.abs(y_true) > 1e-30) & (np.abs(y_pred) > 1e-30)
    if int(mask.sum()) < 2:
        return float("inf")
    return float(np.sqrt(np.mean((np.log(np.abs(y_true[mask])) - np.log(np.abs(y_pred[mask]))) ** 2)))


def _effective_linear_term(coef: float, basis: np.ndarray, target: np.ndarray, rel: float = 1e-8) -> bool:
    """Keep only terms that have measurable scale in the fitted law.

    Least-squares fits often leave tiny numerical intercepts or nearly-zero
    terms in otherwise exact symbolic laws.  Dropping them is an MDL
    simplification: the shorter expression is preferred only when the term's
    maximum contribution is negligible relative to the target scale.
    """

    coef = float(coef)
    if abs(coef) <= 1e-12:
        return False
    basis = np.asarray(basis, dtype=float)
    target = np.asarray(target, dtype=float)
    if basis.size == 0 or not np.all(np.isfinite(basis)):
        return False
    scale = max(float(np.nanmax(np.abs(target))), 1e-30)
    contribution = abs(coef) * max(float(np.nanmax(np.abs(basis))), 1.0)
    return contribution > rel * scale


def _fit_loglinear(features: list[np.ndarray], y: np.ndarray) -> tuple[np.ndarray, float, float] | None:
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(y) & (np.abs(y) > 1e-30)
    logs = []
    for f in features:
        f = np.asarray(f, dtype=float)
        mask = mask & np.isfinite(f) & (np.abs(f) > 1e-30)
        logs.append(np.log(np.abs(f)))
    if int(mask.sum()) < len(features) + 2:
        return None
    X = np.stack([g[mask] for g in logs] + [np.ones(int(mask.sum()))], axis=1)
    target = np.log(np.abs(y[mask]))
    coef, *_ = np.linalg.lstsq(X, target, rcond=None)
    pred = X @ coef
    loss = float(np.sqrt(np.mean((target - pred) ** 2)))
    return coef[:-1], float(np.exp(coef[-1])), loss


def _snap_exp(x: float) -> float:
    grid = [
        -4.0,
        -3.0,
        -math.e,
        -2.6,
        -2.5,
        -2.0,
        -1.5,
        -1.3,
        -1.0,
        -0.5,
        0.0,
        0.5,
        1.0,
        1.3,
        1.5,
        2.0,
        2.5,
        2.6,
        math.e,
        3.0,
        4.0,
    ]
    best = min(grid, key=lambda g: abs(g - x))
    return best if abs(best - x) < 0.08 else float(x)


def _fmt_num(x: float) -> str:
    return f"{float(x):.12g}"


def _expr_power(expr: str, exp: float) -> str:
    exp = _snap_exp(exp)
    if abs(exp - 1.0) < 1e-12:
        return f"({expr})"
    if abs(exp) < 1e-12:
        return "1.0"
    return f"({expr})**{_fmt_num(exp)}"


def _safe_feature(expr: str, arr: np.ndarray) -> tuple[str, np.ndarray] | None:
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    if float(np.nanstd(arr)) <= 1e-12:
        return None
    return expr, arr


def _small_additive_basis(rows: list[dict[str, float]], params: list[str]) -> list[tuple[str, np.ndarray]]:
    """Generic basis terms for additive scientific laws.

    This is intentionally domain-agnostic.  It proposes low-complexity numeric
    measurements from variable ranges: powers for positive coordinates and
    sinusoidal charts for bounded/angle-like coordinates.  Selection happens
    only by train/held-out error, so the basis library is not a benchmark
    answer key.
    """

    arrays = {name: np.asarray([row[name] for row in rows], dtype=float) for name in params}
    out: list[tuple[str, np.ndarray]] = []
    seen: set[str] = set()

    def add(expr: str, arr: np.ndarray):
        if expr in seen:
            return
        item = _safe_feature(expr, arr)
        if item is not None:
            seen.add(expr)
            out.append(item)

    def pow_if_valid(arr: np.ndarray, exp: float) -> np.ndarray | None:
        arr = np.asarray(arr, dtype=float)
        if abs(exp - round(exp)) < 1e-12 and exp >= 0:
            val = arr ** int(round(exp))
        elif exp < 0:
            if np.any(np.abs(arr) <= 1e-12):
                return None
            val = arr ** exp
        else:
            if np.any(arr <= 0):
                return None
            val = arr ** exp
        if not np.all(np.isfinite(val)):
            return None
        return val

    power_grid = [-4.0, -10.0 / 3.0, -3.0, -math.e, -2.5, -2.0, -1.0, -0.5,
                  0.5, 1.0, 1.5, 2.0, 2.5, math.e, 3.0, 3.4, 4.0]
    simple_grid = [-2.0, -1.0, 1.0, 2.0]
    for p in params:
        x = arrays[p]
        if np.all(np.isfinite(x)) and np.all(x > 0):
            for e in power_grid:
                add(_expr_power(p, e), x ** e)

    angle_like = []
    for p in params:
        x = arrays[p]
        name = p.lower()
        lexical_angle = any(tok in name for tok in ("theta", "angle", "phase"))
        bounded_angle = float(np.nanmin(x)) >= -1e-9 and float(np.nanmax(x)) <= math.pi + 1e-9
        if np.all(np.isfinite(x)) and (lexical_angle or bounded_angle):
            angle_like.append(p)
    for p in angle_like:
        x = arrays[p]
        sinx = np.sin(x)
        cosx = np.cos(x)
        add(f"math.sin({p})", sinx)
        add(f"math.cos({p})", cosx)
        for se in [1.0, 2.0, 3.0, math.e]:
            val = pow_if_valid(sinx, se)
            if val is not None:
                add(_expr_power(f"math.sin({p})", se), val)
            val = pow_if_valid(cosx, se)
            if val is not None:
                add(_expr_power(f"math.cos({p})", se), val)
        for se in [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0, math.e]:
            sval = pow_if_valid(sinx, se)
            if sval is None:
                continue
            for ce in [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0, math.e]:
                cval = pow_if_valid(cosx, ce)
                if cval is None:
                    continue
                expr = f"{_expr_power(f'math.sin({p})', se)} * {_expr_power(f'math.cos({p})', ce)}"
                add(expr, sval * cval)

    # Cross a one-dimensional basis with one simple scale coordinate.  This is
    # enough to express laws like x^2 + x/sqrt(z) and I0 * trig(theta)^p while
    # keeping the search small.
    base_terms = list(out)
    for scale in params:
        if scale in angle_like:
            continue
        sx = arrays[scale]
        if not (np.all(np.isfinite(sx)) and np.all(sx > 0)):
            continue
        for sp in [1.0, -1.0, 0.5, -0.5, 2.0]:
            scale_expr = _expr_power(scale, sp)
            scale_arr = sx ** sp
            for expr, arr in base_terms:
                if expr == scale_expr or scale in expr:
                    continue
                add(f"{scale_expr} * {expr}", scale_arr * arr)
    return out


def _fit_additive_basis_candidates(points: list[NBForcePoint], params: list[str]) -> list[LawCandidate]:
    rows = [p.inputs for p in points]
    y = np.asarray([p.force for p in points], dtype=float)
    n = len(y)
    if n < 10:
        return []
    split = max(6, int(round(0.7 * n)))
    basis = _small_additive_basis(rows, params)
    if not basis:
        return []
    train_y = y[:split]
    hold_y = y[split:]
    selected: list[int] = []
    candidates: list[LawCandidate] = []
    residual = train_y.copy()

    arrays = [arr for _expr, arr in basis]
    for _ in range(min(5, max(1, split - 2))):
        best_i = None
        best_loss = float("inf")
        for i, arr in enumerate(arrays):
            if i in selected:
                continue
            X = np.stack([arrays[j][:split] for j in selected + [i]] + [np.ones(split)], axis=1)
            if not np.all(np.isfinite(X)):
                continue
            try:
                coef, *_ = np.linalg.lstsq(X, train_y, rcond=None)
                pred = X @ coef
            except Exception:
                continue
            loss = float(np.sqrt(np.mean((np.log(np.abs(train_y) + 1e-30) - np.log(np.abs(pred) + 1e-30)) ** 2)))
            if loss < best_loss:
                best_i = i
                best_loss = loss
        if best_i is None:
            break
        selected.append(best_i)
        Xtr = np.stack([arrays[j][:split] for j in selected] + [np.ones(split)], axis=1)
        try:
            coef, *_ = np.linalg.lstsq(Xtr, train_y, rcond=None)
        except Exception:
            continue
        pred_tr = Xtr @ coef
        residual = train_y - pred_tr
        Xho = np.stack([arrays[j][split:] for j in selected] + [np.ones(n - split)], axis=1)
        pred_ho = Xho @ coef
        holdout_loss = _safe_log_loss(hold_y, pred_ho)
        if not math.isfinite(holdout_loss):
            continue
        terms = []
        form_terms = []
        for j, c in zip(selected, coef[:-1]):
            if not _effective_linear_term(float(c), arrays[j], train_y):
                continue
            expr = basis[j][0]
            terms.append(f"({_fmt_num(float(c))}) * ({expr})")
            form_terms.append(f"{_fmt_num(float(c))}*{expr}")
        intercept = float(coef[-1])
        if _effective_linear_term(intercept, np.ones_like(train_y), train_y):
            terms.append(_fmt_num(intercept))
            form_terms.append(_fmt_num(intercept))
        if not terms:
            continue
        body = "return " + " + ".join(terms)
        train_loss = _safe_log_loss(train_y, pred_tr)
        complexity = 2.0 + len(selected)
        candidates.append(
            LawCandidate(
                kind=f"additive_basis_{len(selected)}",
                form=" + ".join(form_terms),
                python_body=body,
                train_loss=train_loss,
                holdout_loss=holdout_loss,
                mdl_score=train_loss + holdout_loss + 0.03 * complexity,
                complexity=complexity,
            )
        )
        if float(np.std(residual)) <= 1e-12:
            break
    return candidates


def _fit_link_additive_candidates(points: list[NBForcePoint], params: list[str]) -> list[LawCandidate]:
    """Induce an outer coordinate where an additive law becomes short.

    The operator is intentionally domain-agnostic.  It asks whether a monotone
    link of the observed target (square, square-root, 2/3 power) admits a much
    shorter additive explanation in input coordinates, then renders the inverse
    link.  This is the Newton instance of the MARS rule "birth a coordinate
    only when it compresses residuals and validates on held-out rows."
    """

    rows = [p.inputs for p in points]
    y = np.asarray([p.force for p in points], dtype=float)
    n = len(y)
    if n < 12:
        return []
    basis = _small_additive_basis(rows, params)
    if not basis:
        return []

    split = max(6, int(round(0.7 * n)))
    out: list[LawCandidate] = []
    link_specs = [
        (
            "outer_sqrt",
            lambda arr: arr ** 2,
            lambda expr: f"math.sqrt(max(0.0, ({expr})))",
            1.6,
        ),
        (
            "outer_square",
            lambda arr: np.sqrt(np.maximum(arr, 0.0)),
            lambda expr: f"(({expr}) ** 2)",
            1.6,
        ),
        (
            "outer_pow_1p5",
            lambda arr: np.maximum(arr, 0.0) ** (2.0 / 3.0),
            lambda expr: f"(max(0.0, ({expr})) ** 1.5)",
            1.8,
        ),
    ]

    arrays = [arr for _expr, arr in basis]
    for link_name, target_fn, inverse_render, link_complexity in link_specs:
        z = np.asarray(target_fn(y), dtype=float)
        if not np.all(np.isfinite(z)) or float(np.std(z)) <= 1e-12:
            continue
        selected: list[int] = []
        for _ in range(min(5, max(1, split - 2))):
            best_i = None
            best_loss = float("inf")
            for i, arr in enumerate(arrays):
                if i in selected:
                    continue
                X = np.stack([arrays[j][:split] for j in selected + [i]] + [np.ones(split)], axis=1)
                if not np.all(np.isfinite(X)):
                    continue
                try:
                    coef, *_ = np.linalg.lstsq(X, z[:split], rcond=None)
                except Exception:
                    continue
                expr_terms = []
                for j, c in zip(selected + [i], coef[:-1]):
                    if _effective_linear_term(float(c), arrays[j], z[:split]):
                        expr_terms.append(f"({_fmt_num(float(c))}) * ({basis[j][0]})")
                if _effective_linear_term(float(coef[-1]), np.ones_like(z[:split]), z[:split]):
                    expr_terms.append(_fmt_num(float(coef[-1])))
                if not expr_terms:
                    continue
                expr = " + ".join(expr_terms)
                body = "return " + inverse_render(expr)
                try:
                    pred = _eval_body(body, params, rows[:split])
                    loss = _safe_log_loss(y[:split], pred)
                except Exception:
                    loss = float("inf")
                if loss < best_loss:
                    best_i = i
                    best_loss = loss
            if best_i is None:
                break
            selected.append(best_i)
            Xtr = np.stack([arrays[j][:split] for j in selected] + [np.ones(split)], axis=1)
            Xho = np.stack([arrays[j][split:] for j in selected] + [np.ones(n - split)], axis=1)
            try:
                coef, *_ = np.linalg.lstsq(Xtr, z[:split], rcond=None)
                pred_z_tr = Xtr @ coef
                pred_z_ho = Xho @ coef
            except Exception:
                continue
            terms = []
            form_terms = []
            for j, c in zip(selected, coef[:-1]):
                if not _effective_linear_term(float(c), arrays[j], z[:split]):
                    continue
                terms.append(f"({_fmt_num(float(c))}) * ({basis[j][0]})")
                form_terms.append(f"{_fmt_num(float(c))}*{basis[j][0]}")
            if _effective_linear_term(float(coef[-1]), np.ones_like(z[:split]), z[:split]):
                terms.append(_fmt_num(float(coef[-1])))
                form_terms.append(_fmt_num(float(coef[-1])))
            if not terms:
                continue
            inner = " + ".join(terms)
            body = "return " + inverse_render(inner)
            try:
                pred_tr = _eval_body(body, params, rows[:split])
                pred_ho = _eval_body(body, params, rows[split:])
            except Exception:
                continue
            train_loss = _safe_log_loss(y[:split], pred_tr)
            holdout_loss = _safe_log_loss(y[split:], pred_ho)
            if not math.isfinite(holdout_loss):
                continue
            complexity = link_complexity + len(selected)
            out.append(
                LawCandidate(
                    kind=f"link_additive_{link_name}_{len(selected)}",
                    form=f"{link_name}^-1(" + " + ".join(form_terms) + ")",
                    python_body=body,
                    train_loss=train_loss,
                    holdout_loss=holdout_loss,
                    mdl_score=train_loss + holdout_loss + 0.03 * complexity,
                    complexity=complexity,
                )
            )
    return out


def _fit_link_power_candidates(points: list[NBForcePoint], params: list[str]) -> list[LawCandidate]:
    """Induce bounded/singular target links followed by compact power laws."""

    rows = [p.inputs for p in points]
    y = np.asarray([p.force for p in points], dtype=float)
    n = len(y)
    if n < 10:
        return []
    arrays = {name: np.asarray([row[name] for row in rows], dtype=float) for name in params}
    feature_arrays = [arrays[p] for p in params]
    out: list[LawCandidate] = []

    def add_candidate(kind: str, z: np.ndarray, render) -> None:
        if not np.all(np.isfinite(z)) or float(np.std(z)) <= 1e-12:
            return
        split = max(6, int(round(0.7 * n)))
        fit = _fit_loglinear([a[:split] for a in feature_arrays], z[:split])
        if fit is None:
            return
        exps, const, train_z_loss = fit
        inner = _fmt_num(const)
        form = _fmt_num(const)
        for p, exp in zip(params, exps):
            snapped = _snap_exp(float(exp))
            if abs(snapped) < 1e-12:
                continue
            inner += f" * {_expr_power(p, snapped)}"
            form += f" * ({p})^{_fmt_num(snapped)}"
        body = "return " + render(inner)
        try:
            pred_ho = _eval_body(body, params, rows[split:])
            pred_tr = _eval_body(body, params, rows[:split])
        except Exception:
            return
        train_loss = _safe_log_loss(y[:split], pred_tr)
        holdout_loss = _safe_log_loss(y[split:], pred_ho)
        if not math.isfinite(holdout_loss):
            return
        complexity = 2.2 + sum(abs(float(e)) > 1e-9 for e in exps)
        out.append(
            LawCandidate(
                kind=kind,
                form=f"{kind}({form})",
                python_body=body,
                train_loss=train_loss,
                holdout_loss=holdout_loss,
                mdl_score=train_loss + holdout_loss + 0.02 * complexity + 0.2 * train_z_loss,
                complexity=complexity,
            )
        )

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        # y = 1 / (exp(z) + 1)  ->  z = log(1/y - 1)
        mask = (y > 1e-12) & (y < 1.0 - 1e-12)
        if int(mask.sum()) >= max(8, len(params) + 3):
            z = np.full_like(y, np.nan, dtype=float)
            z[mask] = np.log((1.0 / y[mask]) - 1.0)
            add_candidate(
                "bounded_exp_denominator_plus",
                z,
                lambda inner: f"1.0 / (math.exp({inner}) + 1.0)",
            )

        # y = 1 / (exp(z) - 1)  ->  z = log(1 + 1/y)
        mask = y > 1e-12
        if int(mask.sum()) >= max(8, len(params) + 3):
            z = np.full_like(y, np.nan, dtype=float)
            z[mask] = np.log1p(1.0 / y[mask])
            add_candidate(
                "singular_exp_denominator_minus",
                z,
                lambda inner: f"1.0 / max(1e-300, math.exp({inner}) - 1.0)",
            )

            # y = 1 / (-log(z) - 1), where z is a compact positive power law.
            u = np.full_like(y, np.nan, dtype=float)
            u[mask] = np.exp(-(1.0 + 1.0 / y[mask]))
            add_candidate(
                "inverse_log_denominator",
                u,
                lambda inner: f"1.0 / max(1e-300, (-math.log(max(1e-300, {inner})) - 1.0))",
            )
    return out


def _fit_asymptotic_lift_candidates(points: list[NBForcePoint], params: list[str]) -> list[LawCandidate]:
    """Lift a local power asymptote into a refutable nonlinear law.

    If observations live on a high-occupation/asymptotic shelf, a law like
    y = 1/(exp(z)-1) looks locally like y ~= 1/z.  A plain fitter therefore
    discovers only y ~= C prod x^a.  This routine treats that as a *clue*:
    it proposes z ~= (1/C) prod x^-a and verifies the lifted exponential
    denominator on held-out rows.
    """

    rows = [p.inputs for p in points]
    y = np.asarray([p.force for p in points], dtype=float)
    n = len(y)
    if n < 10 or not np.all(np.isfinite(y)):
        return []
    if float(np.nanmedian(np.abs(y))) < 10.0:
        return []
    arrays = {name: np.asarray([row[name] for row in rows], dtype=float) for name in params}
    feature_arrays = [arrays[p] for p in params]
    split = max(6, int(round(0.7 * n)))
    fit = _fit_loglinear([a[:split] for a in feature_arrays], y[:split])
    if fit is None:
        return []
    exps, const, train_power_loss = fit
    if not math.isfinite(const) or const <= 0:
        return []

    inner = _fmt_num(1.0 / const)
    form = _fmt_num(1.0 / const)
    active = 0
    for p, exp in zip(params, exps):
        snapped = _snap_exp(-float(exp))
        if abs(snapped) < 1e-12:
            continue
        active += 1
        inner += f" * {_expr_power(p, snapped)}"
        form += f" * ({p})^{_fmt_num(snapped)}"
    if active == 0:
        return []

    def safe_minus(inner_expr: str) -> str:
        return (
            f"(0.0 if ({inner_expr}) > 700.0 else "
            f"1.0 / max(1e-300, math.exp({inner_expr}) - 1.0))"
        )

    out: list[LawCandidate] = []
    for kind, render, complexity_bonus in [
        ("asymptotic_lift_exp_minus", safe_minus, 2.6),
    ]:
        body = "return " + render(inner)
        try:
            pred_tr = _eval_body(body, params, rows[:split])
            pred_ho = _eval_body(body, params, rows[split:])
        except Exception:
            continue
        train_loss = _safe_log_loss(y[:split], pred_tr)
        holdout_loss = _safe_log_loss(y[split:], pred_ho)
        if not math.isfinite(holdout_loss):
            continue
        complexity = complexity_bonus + active
        out.append(
            LawCandidate(
                kind=kind,
                form=f"1/(exp({form})-1)",
                python_body=body,
                train_loss=train_loss,
                holdout_loss=holdout_loss,
                mdl_score=train_loss + holdout_loss + 0.015 * complexity + 0.08 * train_power_loss,
                complexity=complexity,
            )
        )
    return out


def _fit_scale_chart_candidates(points: list[NBForcePoint], params: list[str]) -> list[LawCandidate]:
    """Fit y = scale * short_sum(chart(coord)) for factorized observations."""

    rows = [p.inputs for p in points]
    y = np.asarray([p.force for p in points], dtype=float)
    n = len(y)
    if n < 10:
        return []
    split = max(6, int(round(0.7 * n)))
    arrays = {name: np.asarray([row[name] for row in rows], dtype=float) for name in params}
    out: list[LawCandidate] = []

    def is_angle(name: str, arr: np.ndarray) -> bool:
        low = name.lower()
        return any(tok in low for tok in ("theta", "angle", "phase")) or (
            np.all(np.isfinite(arr))
            and float(np.nanmin(arr)) >= -1e-9
            and float(np.nanmax(arr)) <= math.pi + 1e-9
        )

    for coord in params:
        x = arrays[coord]
        if not is_angle(coord, x):
            continue
        sinx = np.sin(x)
        cosx = np.cos(x)
        chart_terms: list[tuple[str, np.ndarray]] = []
        for se in [0.0, 1.0, 2.0, 3.0, math.e, -1.0, -2.0, -3.0]:
            sval = np.ones_like(x) if abs(se) < 1e-12 else None
            if sval is None:
                if abs(se - round(se)) < 1e-12 and se >= 0:
                    sval = sinx ** int(round(se))
                elif se < 0 and np.all(np.abs(sinx) > 1e-12):
                    sval = sinx ** se
                elif se > 0 and np.all(sinx > 0):
                    sval = sinx ** se
            if sval is None:
                continue
            for ce in [0.0, 1.0, 2.0, 3.0, math.e, -1.0, -2.0, -3.0]:
                if abs(se) < 1e-12 and abs(ce) < 1e-12:
                    continue
                cval = np.ones_like(x) if abs(ce) < 1e-12 else None
                if cval is None:
                    if abs(ce - round(ce)) < 1e-12 and ce >= 0:
                        cval = cosx ** int(round(ce))
                    elif ce < 0 and np.all(np.abs(cosx) > 1e-12):
                        cval = cosx ** ce
                    elif ce > 0 and np.all(cosx > 0):
                        cval = cosx ** ce
                if cval is None:
                    continue
                expr_parts = []
                if abs(se) > 1e-12:
                    expr_parts.append(_expr_power(f"math.sin({coord})", se))
                if abs(ce) > 1e-12:
                    expr_parts.append(_expr_power(f"math.cos({coord})", ce))
                expr = " * ".join(expr_parts) if expr_parts else "1.0"
                arr = sval * cval
                if np.all(np.isfinite(arr)) and np.std(arr) > 1e-12:
                    chart_terms.append((expr, arr))

        for scale in params:
            if scale == coord:
                continue
            s = arrays[scale]
            if not (np.all(np.isfinite(s)) and np.all(s > 0)):
                continue
            terms = [(f"({scale}) * {expr}", s * arr) for expr, arr in chart_terms]

            def add_linear_candidate(indices: list[int], kind: str) -> None:
                if not indices:
                    return
                Xtr = np.stack([terms[j][1][:split] for j in indices], axis=1)
                Xho = np.stack([terms[j][1][split:] for j in indices], axis=1)
                try:
                    coef, *_ = np.linalg.lstsq(Xtr, y[:split], rcond=None)
                    pred_tr = Xtr @ coef
                    pred_ho = Xho @ coef
                except Exception:
                    return
                train_loss = _safe_log_loss(y[:split], pred_tr)
                holdout_loss = _safe_log_loss(y[split:], pred_ho)
                if not math.isfinite(holdout_loss):
                    return
                body_terms = []
                form_terms = []
                for j, c in zip(indices, coef):
                    if not _effective_linear_term(float(c), terms[j][1][:split], y[:split]):
                        continue
                    body_terms.append(f"({_fmt_num(float(c))}) * ({terms[j][0]})")
                    form_terms.append(f"{_fmt_num(float(c))}*{terms[j][0]}")
                if not body_terms:
                    return
                out.append(
                    LawCandidate(
                        kind=kind,
                        form=" + ".join(form_terms),
                        python_body="return " + " + ".join(body_terms),
                        train_loss=train_loss,
                        holdout_loss=holdout_loss,
                        mdl_score=train_loss + holdout_loss + 0.02 * (2 + len(indices)),
                        complexity=2 + len(indices),
                    )
                )

            # Single chart monomials are valuable for ratio laws; a quadratic
            # chart is the minimal representation of squared latent projections.
            for i in range(len(terms)):
                add_linear_candidate([i], "scale_chart_monomial")
            term_index = {expr: i for i, (expr, _arr) in enumerate(terms)}
            quad = [
                term_index.get(f"({scale}) * {_expr_power(f'math.sin({coord})', 2.0)}"),
                term_index.get(f"({scale}) * {_expr_power(f'math.sin({coord})', 1.0)} * {_expr_power(f'math.cos({coord})', 1.0)}"),
                term_index.get(f"({scale}) * {_expr_power(f'math.cos({coord})', 2.0)}"),
            ]
            if all(i is not None for i in quad):
                add_linear_candidate([int(i) for i in quad], "scale_chart_quadratic")

            selected: list[int] = []
            for _ in range(min(5, max(1, split - 1))):
                best = None
                for i, (_expr, arr) in enumerate(terms):
                    if i in selected:
                        continue
                    Xtr = np.stack([terms[j][1][:split] for j in selected + [i]], axis=1)
                    try:
                        coef, *_ = np.linalg.lstsq(Xtr, y[:split], rcond=None)
                        pred_tr = Xtr @ coef
                    except Exception:
                        continue
                    Xho = np.stack([terms[j][1][split:] for j in selected + [i]], axis=1)
                    pred_ho = Xho @ coef
                    loss = _safe_log_loss(y[split:], pred_ho)
                    train_loss = _safe_log_loss(y[:split], pred_tr)
                    score = loss + 0.02 * (len(selected) + 1)
                    if best is None or score < best[0]:
                        best = (score, i, coef, train_loss, loss)
                if best is None:
                    break
                _score, i, coef, train_loss, loss = best
                selected.append(i)
                body_terms = []
                form_terms = []
                for j, c in zip(selected, coef):
                    if not _effective_linear_term(float(c), terms[j][1][:split], y[:split]):
                        continue
                    body_terms.append(f"({_fmt_num(float(c))}) * ({terms[j][0]})")
                    form_terms.append(f"{_fmt_num(float(c))}*{terms[j][0]}")
                if body_terms:
                    out.append(
                        LawCandidate(
                            kind=f"scale_chart_additive_{len(selected)}",
                            form=" + ".join(form_terms),
                            python_body="return " + " + ".join(body_terms),
                            train_loss=train_loss,
                            holdout_loss=loss,
                            mdl_score=train_loss + loss + 0.025 * (2 + len(selected)),
                            complexity=2 + len(selected),
                        )
                    )
    return out


def _fit_sparse_latent_additive_candidates(points: list[NBForcePoint], params: list[str]) -> list[LawCandidate]:
    """Induce a small latent equation before rendering the final law.

    Many scientific laws are not a single monomial; they become simple only
    after a target-side link exposes a sparse additive latent variable:

        link(y) = c0 + c1*f1(x) + c2*f2(x) + ...

    The features are still born from the generic basis library, and every
    candidate is verified on held-out rows.  This gives the weak model a real
    hypothesis generator for sums/differences without naming any benchmark.
    """

    rows = [p.inputs for p in points]
    y = np.asarray([p.force for p in points], dtype=float)
    n = len(y)
    if n < 10:
        return []
    basis = _small_additive_basis(rows, params)
    if not basis:
        return []
    split = max(6, int(round(0.7 * n)))
    out: list[LawCandidate] = []
    link_specs = [
        ("identity", lambda arr: arr, lambda expr: f"({expr})", 1.2),
        ("outer_sqrt", lambda arr: arr ** 2, lambda expr: f"math.sqrt(max(0.0, ({expr})))", 1.6),
        ("outer_square", lambda arr: np.sqrt(np.maximum(arr, 0.0)), lambda expr: f"(({expr}) ** 2)", 1.6),
        (
            "outer_pow_1p5",
            lambda arr: np.maximum(arr, 0.0) ** (2.0 / 3.0),
            lambda expr: f"(max(0.0, ({expr})) ** 1.5)",
            1.8,
        ),
    ]

    import itertools

    def render_sum(indices: tuple[int, ...], coef: np.ndarray) -> tuple[str, list[str]]:
        terms = []
        form_terms = []
        for j, c in zip(indices, coef[:-1]):
            cf = float(c)
            if not _effective_linear_term(cf, arrays[j], z[:split]):
                continue
            terms.append(f"({_fmt_num(cf)}) * ({basis[j][0]})")
            form_terms.append(f"{_fmt_num(cf)}*{basis[j][0]}")
        intercept = float(coef[-1])
        if _effective_linear_term(intercept, np.ones_like(z[:split]), z[:split]):
            terms.append(_fmt_num(intercept))
            form_terms.append(_fmt_num(intercept))
        if not terms:
            return "0.0", []
        return " + ".join(terms), form_terms

    arrays = [np.asarray(arr, dtype=float) for _expr, arr in basis]
    for link_name, target_fn, inverse_render, link_complexity in link_specs:
        try:
            z = np.asarray(target_fn(y), dtype=float)
        except Exception:
            continue
        if not np.all(np.isfinite(z)) or float(np.std(z)) <= 1e-12:
            continue
        ztr = z[:split]
        scale = float(np.std(ztr)) + 1e-12
        ranked: list[tuple[float, int]] = []
        for i, arr in enumerate(arrays):
            if not np.all(np.isfinite(arr)):
                continue
            X = np.stack([arr[:split], np.ones(split)], axis=1)
            try:
                coef, *_ = np.linalg.lstsq(X, ztr, rcond=None)
                pred = X @ coef
            except Exception:
                continue
            loss = float(np.sqrt(np.mean((ztr - pred) ** 2)) / scale)
            if math.isfinite(loss):
                ranked.append((loss, i))
        if not ranked:
            continue
        ranked.sort()
        keep = [i for _loss, i in ranked[: min(30, len(ranked))]]
        seen_combos: set[tuple[int, ...]] = set()
        for size in (1, 2, 3):
            for combo in itertools.combinations(keep, size):
                combo = tuple(sorted(combo))
                if combo in seen_combos:
                    continue
                seen_combos.add(combo)
                Xtr = np.stack([arrays[j][:split] for j in combo] + [np.ones(split)], axis=1)
                Xho = np.stack([arrays[j][split:] for j in combo] + [np.ones(n - split)], axis=1)
                if not np.all(np.isfinite(Xtr)) or not np.all(np.isfinite(Xho)):
                    continue
                try:
                    coef, *_ = np.linalg.lstsq(Xtr, ztr, rcond=None)
                    pred_z_tr = Xtr @ coef
                except Exception:
                    continue
                latent_loss = float(np.sqrt(np.mean((ztr - pred_z_tr) ** 2)) / scale)
                if not math.isfinite(latent_loss):
                    continue
                inner, form_terms = render_sum(combo, coef)
                if not form_terms:
                    continue
                body = "return " + inverse_render(inner)
                try:
                    pred_tr = _eval_body(body, params, rows[:split])
                    pred_ho = _eval_body(body, params, rows[split:])
                except Exception:
                    continue
                train_loss = _safe_log_loss(y[:split], pred_tr)
                holdout_loss = _safe_log_loss(y[split:], pred_ho)
                if not math.isfinite(holdout_loss):
                    continue
                complexity = link_complexity + 0.8 * size
                out.append(
                    LawCandidate(
                        kind=f"sparse_latent_{link_name}_{size}",
                        form=f"{link_name}^-1(" + " + ".join(form_terms) + ")",
                        python_body=body,
                        train_loss=train_loss,
                        holdout_loss=holdout_loss,
                        mdl_score=train_loss + holdout_loss + 0.03 * complexity + 0.10 * latent_loss,
                        complexity=complexity,
                    )
                )
    return out


def _eval_body(body: str, params: list[str], rows: list[dict[str, float]]) -> np.ndarray:
    code = f"def __law({', '.join(params)}):\n    {body}"
    ns: dict[str, Any] = {
        "__builtins__": {"abs": abs, "min": min, "max": max, "pow": pow},
        "math": math,
    }
    exec(code, ns)  # noqa: S102
    fn = ns["__law"]
    out = []
    for row in rows:
        out.append(float(fn(**{p: row[p] for p in params})))
    return np.asarray(out, dtype=float)


def _candidate_from_features(
    *,
    kind: str,
    feature_exprs: list[str],
    feature_arrays: list[np.ndarray],
    rows: list[dict[str, float]],
    forces: np.ndarray,
    params: list[str],
    complexity: float,
) -> LawCandidate | None:
    n = len(forces)
    if n < 5:
        return None
    split = max(3, int(round(0.75 * n)))
    fit = _fit_loglinear([f[:split] for f in feature_arrays], forces[:split])
    if fit is None:
        return None
    exps, const, train_loss = fit
    body = "return " + _fmt_num(const)
    form = _fmt_num(const)
    for expr, exp in zip(feature_exprs, exps):
        snapped = _snap_exp(float(exp))
        if abs(snapped) < 1e-12:
            continue
        body += f" * {_expr_power(expr, snapped)}"
        form += f" * ({expr})^{_fmt_num(snapped)}"
    try:
        pred = _eval_body(body, params, rows[split:])
        holdout_loss = _safe_log_loss(forces[split:], pred)
    except Exception:
        holdout_loss = float("inf")
    mdl_score = train_loss + holdout_loss + 0.01 * complexity
    return LawCandidate(
        kind=kind,
        form=form,
        python_body=body,
        train_loss=train_loss,
        holdout_loss=holdout_loss,
        mdl_score=mdl_score,
        complexity=complexity,
    )


def fit_law_candidates(points: list[NBForcePoint], params: list[str]) -> list[LawCandidate]:
    rows = [p.inputs for p in points]
    forces = np.asarray([p.force for p in points], dtype=float)
    arrays = {name: np.asarray([row[name] for row in rows], dtype=float) for name in params}

    candidates: list[LawCandidate] = []

    # Generic separable law: C * prod x_i^a_i.
    c = _candidate_from_features(
        kind="separable",
        feature_exprs=params,
        feature_arrays=[arrays[p] for p in params],
        rows=rows,
        forces=forces,
        params=params,
        complexity=1.0 + len(params),
    )
    if c is not None:
        candidates.append(c)

    # Pair bases over the first two signature variables, with remaining
    # variables as multiplicative factors. This captures laws such as
    # (m1+m2)^2/r^1.5 and (m1^2+m2^2)*r^2.
    if len(params) >= 2:
        p1, p2 = params[0], params[1]
        extras = params[2:]
        pair_bases = {
            "pair_sum": (f"{p1}+{p2}", arrays[p1] + arrays[p2], 1.2),
            "pair_product": (f"{p1}*{p2}", arrays[p1] * arrays[p2], 1.2),
            "pair_sq_sum": (f"{p1}**2+{p2}**2", arrays[p1] ** 2 + arrays[p2] ** 2, 1.5),
            "pair_sq_product": (f"{p1}**2*{p2}**2", (arrays[p1] ** 2) * (arrays[p2] ** 2), 1.6),
            "pair_product_sum": (
                f"({p1}*{p2})*({p1}+{p2})",
                (arrays[p1] * arrays[p2]) * (arrays[p1] + arrays[p2]),
                1.8,
            ),
        }
        for kind, (expr, arr, base_complexity) in pair_bases.items():
            feature_exprs = [expr] + extras
            feature_arrays = [arr] + [arrays[e] for e in extras]
            c = _candidate_from_features(
                kind=kind,
                feature_exprs=feature_exprs,
                feature_arrays=feature_arrays,
                rows=rows,
                forces=forces,
                params=params,
                complexity=base_complexity + len(extras),
            )
            if c is not None:
                candidates.append(c)

        # Composite grammar: pair interaction plus individual variables. This
        # lets the layer discover asymmetric laws such as q1^3*q2/r^2 and
        # q2^2*(q1+q2)^3/r^2 without benchmark-specific branches.
        composite_sets = {
            "pair_sum_plus_individuals": (
                [f"{p1}+{p2}", p1, p2] + extras,
                [arrays[p1] + arrays[p2], arrays[p1], arrays[p2]] + [arrays[e] for e in extras],
                2.5 + len(extras),
            ),
            "pair_product_plus_individuals": (
                [f"{p1}*{p2}", p1, p2] + extras,
                [arrays[p1] * arrays[p2], arrays[p1], arrays[p2]] + [arrays[e] for e in extras],
                2.5 + len(extras),
            ),
        }
        for kind, (feature_exprs, feature_arrays, complexity) in composite_sets.items():
            c = _candidate_from_features(
                kind=kind,
                feature_exprs=feature_exprs,
                feature_arrays=feature_arrays,
                rows=rows,
                forces=forces,
                params=params,
                complexity=complexity,
            )
            if c is not None:
                candidates.append(c)

    candidates.extend(_fit_additive_basis_candidates(points, params))
    candidates.extend(_fit_link_additive_candidates(points, params))
    candidates.extend(_fit_sparse_latent_additive_candidates(points, params))
    candidates.extend(_fit_link_power_candidates(points, params))
    candidates.extend(_fit_asymptotic_lift_candidates(points, params))
    candidates.extend(_fit_scale_chart_candidates(points, params))

    candidates.sort(key=lambda c: (c.mdl_score, c.holdout_loss, c.complexity))
    return candidates


def build_force_analyzers(system: str) -> list[ProgramHypothesis]:
    """Analyzer hypotheses from raw NewtonBench observation to force target."""

    def scalar_passthrough(_current, ctx):
        return float(ctx["raw_output"])

    def simple_from_velocity(_current, ctx):
        raw = ctx["raw_output"]
        mass2 = float(ctx["inputs"].get("mass2", ctx["inputs"].get("m2", 1.0)))
        t = _float_array(raw["time"])
        v = _float_array(raw["velocity"])
        if len(t) < 2 or len(v) < 2:
            raise ValueError("need >=2 velocity samples")
        acc = (v[1] - v[0]) / max(t[1] - t[0], 1e-12)
        return abs(mass2 * acc)

    def complex_from_velocity(_current, ctx):
        raw = ctx["raw_output"]
        mass2 = float(ctx["inputs"].get("mass2", ctx["inputs"].get("m2", 1.0)))
        t = _float_array(raw["time"])
        v = _velocity_array(raw["velocity"])
        if len(t) < 2 or len(v) < 2:
            raise ValueError("need >=2 velocity samples")
        acc_vec = (v[1] - v[0]) / max(t[1] - t[0], 1e-12)
        return float(mass2 * np.linalg.norm(acc_vec))

    if system == "vanilla_equation":
        return [
            ProgramHypothesis(
                name="scalar_passthrough",
                description="Use scalar observation as the direct law target.",
                fn=scalar_passthrough,
                complexity=0.2,
                causal_anchors=("raw_output",),
            )
        ]
    if system == "simple_system":
        return [
            ProgramHypothesis(
                name="simple_velocity_acceleration",
                description="Recover force from 1D velocity finite difference: |m2 * dv/dt|.",
                fn=simple_from_velocity,
                complexity=1.0,
                causal_anchors=("time", "velocity", "mass2"),
            )
        ]
    if system == "complex_system":
        return [
            ProgramHypothesis(
                name="complex_velocity_acceleration",
                description="Recover force from 2D velocity finite difference: m2 * ||dv/dt||.",
                fn=complex_from_velocity,
                complexity=1.2,
                causal_anchors=("time", "velocity", "mass2"),
            )
        ]
    return []


def recover_force(system: str, inputs: dict[str, float], raw_output: Any) -> NBForcePoint | None:
    analyzers = build_force_analyzers(system)
    traces = [
        RuleTrace(
            current=None,
            target=0.0,
            context={"inputs": inputs, "raw_output": raw_output},
        )
    ]
    ranked = rank_hypotheses(analyzers, traces, complexity_weight=0.0)
    # `target` is a dummy, so rank by explicit order. The call above still gives
    # us the same ProgramHypothesis interface as the rest of CPI.
    for hyp, _score in ranked:
        try:
            force = float(hyp.fn(None, {"inputs": inputs, "raw_output": raw_output}))
            if math.isfinite(force) and force > 0:
                clean_inputs = {
                    k: float(inputs[k])
                    for k in inputs
                    if k not in {"initial_velocity", "duration", "time_step", "m1", "m2"}
                }
                return NBForcePoint(
                    inputs=clean_inputs,
                    force=force,
                    raw_output=raw_output,
                    analyzer=hyp.name,
                )
        except Exception:
            continue
    return None


def make_probe_inputs(params: list[str], *, system: str) -> list[dict[str, float]]:
    """A small deterministic design matrix with signature-order interventions."""
    base = {p: 10.0 for p in params}
    if "distance" in base:
        base["distance"] = 2.0
    values = {
        "mass1": [3.0, 10.0, 30.0, 100.0],
        "mass2": [4.0, 12.0, 40.0, 120.0],
        "distance": [1.0, 2.0, 4.0, 8.0],
    }
    rows: list[dict[str, float]] = []
    for p in params:
        for v in values.get(p, [1.0, 3.0, 10.0, 30.0]):
            row = dict(base)
            row[p] = float(v)
            rows.append(row)
    combos = [
        (3.0, 4.0, 1.0),
        (10.0, 40.0, 2.0),
        (30.0, 12.0, 4.0),
        (100.0, 120.0, 8.0),
        (100.0, 4.0, 2.0),
        (3.0, 120.0, 4.0),
    ]
    if {"mass1", "mass2", "distance"}.issubset(set(params)):
        for m1, m2, r in combos:
            rows.append({"mass1": m1, "mass2": m2, "distance": r})

    if system in {"simple_system", "complex_system"}:
        for row in rows:
            row["initial_velocity"] = 0.0
            row["duration"] = 0.25 if system == "complex_system" else 0.05
            row["time_step"] = 0.01
    return rows


def run_nb_cpi_task(
    *,
    module_name: str,
    difficulty: str,
    law_version: str,
    system: str,
    judge_model: str = "gpt41",
    noise_level: float = 0.0,
) -> dict[str, Any]:
    module = importlib.import_module(f"modules.{module_name}")
    signature = str(module.FUNCTION_SIGNATURE).strip()
    params = signature[signature.index("(") + 1 : signature.rindex(")")].split(",")
    params = [p.strip() for p in params if p.strip()]

    points: list[NBForcePoint] = []
    raw_rows: list[dict[str, Any]] = []
    for inp in make_probe_inputs(params, system=system):
        raw = module.run_experiment_for_module(
            noise_level=noise_level,
            difficulty=difficulty,
            system=system,
            law_version=law_version,
            **inp,
        )
        point = recover_force(system, inp, raw)
        raw_rows.append({"inputs": inp, "raw_output": raw, "point": point})
        if point is not None:
            points.append(point)

    cands = fit_law_candidates(points, params)
    best = cands[0] if cands else None
    law_code = best.function_code(signature) if best is not None else ""
    try:
        param_desc = getattr(module, "PARAM_DESCRIPTION", "")
        ev = module.evaluate_law(
            law_code,
            param_description=param_desc,
            difficulty=difficulty,
            law_version=law_version,
            judge_model_name=judge_model,
        )
    except Exception as e:
        ev = {
            "rmsle": float("nan"),
            "exact_accuracy": 0.0,
            "symbolic_equivalent": False,
            "symbolic_msg": f"eval error: {type(e).__name__}: {e}",
        }

    return {
        "module": module_name,
        "difficulty": difficulty,
        "law_version": law_version,
        "system": system,
        "noise_level": noise_level,
        "n_points": len(points),
        "analyzers": sorted({p.analyzer for p in points}),
        "best": None
        if best is None
        else {
            "kind": best.kind,
            "form": best.form,
            "python_body": best.python_body,
            "train_loss": best.train_loss,
            "holdout_loss": best.holdout_loss,
            "mdl_score": best.mdl_score,
            "complexity": best.complexity,
        },
        "candidates": [
            {
                "kind": c.kind,
                "form": c.form,
                "train_loss": c.train_loss,
                "holdout_loss": c.holdout_loss,
                "mdl_score": c.mdl_score,
            }
            for c in cands[:6]
        ],
        "submitted": law_code,
        "SA": float(ev.get("exact_accuracy", 0.0)),
        "rmsle": ev.get("rmsle", float("nan")),
        "numerical_accuracy": float(math.exp(-ev.get("rmsle", float("inf"))))
        if ev.get("rmsle", float("nan")) == ev.get("rmsle", float("nan"))
        else 0.0,
        "symbolic_equivalent": bool(ev.get("symbolic_equivalent", False)),
        "symbolic_msg": str(ev.get("symbolic_msg", ""))[:500],
    }
