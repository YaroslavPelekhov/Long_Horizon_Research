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
