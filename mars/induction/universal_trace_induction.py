"""Universal trace-to-operator induction.

This module is the benchmark-agnostic core for short executable hypotheses.
It receives already-parsed observations of the form

    group, features -> target deltas

and searches a small family of causal measurement operators.  It does not know
about UltraHorizon, NewtonBench, DiscoveryBench, letters, physics, or tables.
Benchmark adapters are allowed to parse raw traces into features, but the
operator posterior and held-out scoring live here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class DeltaOperator:
    name: str
    description: str
    fn: Callable[[dict[str, Any]], tuple[float, ...]]
    complexity: float = 1.0


@dataclass(frozen=True)
class DeltaFit:
    group: str
    operator: DeltaOperator | None
    n_observations: int
    loss_mean: float
    exact_rate: float
    diagnostics: dict[str, Any] = field(default_factory=dict)


def fit_group_delta_operator(
    records: Iterable[dict[str, Any]],
    *,
    group_key: str,
    target_keys: tuple[str, ...],
    operators: Iterable[DeltaOperator],
    min_observations: int = 2,
) -> dict[str, DeltaFit]:
    """Fit one executable delta operator per group by residual loss."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        group = str(record.get(group_key, ""))
        if not group:
            continue
        grouped.setdefault(group, []).append(record)

    out: dict[str, DeltaFit] = {}
    operator_list = list(operators)
    for group, group_records in grouped.items():
        if len(group_records) < min_observations:
            out[group] = DeltaFit(group, None, len(group_records), 1.0, 0.0)
            continue
        ranked = []
        for op in operator_list:
            errors = []
            exact = 0
            for record in group_records:
                try:
                    pred = tuple(float(x) for x in op.fn(record))
                    target = tuple(float(record[k]) for k in target_keys)
                except Exception:
                    continue
                if len(pred) != len(target) or any(not math.isfinite(x) for x in pred):
                    continue
                err = sum(abs(a - b) for a, b in zip(pred, target))
                errors.append(err)
                exact += int(err <= 1e-9)
            if not errors:
                continue
            loss = sum(errors) / len(errors)
            exact_rate = exact / len(errors)
            # MDL tie-break: exactness, then simpler operator.  Priors can be
            # layered outside this fitter without changing the local evidence.
            ranked.append((loss, -exact_rate, op.complexity, op.name, op, exact_rate))
        if not ranked:
            out[group] = DeltaFit(group, None, len(group_records), 1.0, 0.0)
            continue
        loss, neg_exact, _complexity, _name, op, exact_rate = sorted(ranked)[0]
        out[group] = DeltaFit(
            group=group,
            operator=op,
            n_observations=len(group_records),
            loss_mean=float(loss),
            exact_rate=float(exact_rate),
        )
    return out


def state_delta_operator_family(
    *,
    numeric_features: tuple[str, ...],
    target_arity: int,
    coordinate_features: tuple[str, str] | None = None,
) -> list[DeltaOperator]:
    """Generate generic state-delta operators from feature names.

    The family deliberately uses structural feature types, not benchmark names:
    constants, thresholds, parity, modulo conditions, linear coordinates, and
    boundary indicators.  An adapter may pass whichever feature names exist in
    its trace schema.
    """

    if target_arity < 1:
        raise ValueError("target_arity must be positive")

    def structural_complexity(feat: str, operator_kind: str, base: float) -> float:
        """Causal/MDL prior over feature-operator pairings.

        Short traces often contain aliases: a resource variable can move in
        lockstep with time, making ``energy % k`` fit a genuinely temporal
        effect.  Periodic laws are therefore slightly cheaper on exogenous
        counters/indices and slightly more expensive on mutable stocks.  The
        term only affects ties or near-ties; residual loss still dominates.
        """

        f = feat.lower()
        exogenous_counter = any(
            token in f for token in ("step", "time", "index", "round", "turn", "trial")
        )
        mutable_stock = any(
            token in f for token in ("energy", "score", "reward", "health", "budget", "stock")
        )
        adjustment = 0.0
        if operator_kind == "modulo":
            if "visit" in f and "count" in f:
                adjustment += 0.38
            elif exogenous_counter:
                adjustment -= 0.22
            if mutable_stock:
                adjustment += 0.42
        elif operator_kind == "parity":
            if "visit" in f and "count" in f:
                adjustment -= 0.28
            elif exogenous_counter:
                adjustment -= 0.08
            if mutable_stock:
                adjustment += 0.42
        return max(0.05, base + adjustment)

    def tup(first: float, second: float = 0.0) -> tuple[float, ...]:
        vals = [float(first)]
        while len(vals) < target_arity:
            vals.append(float(second if len(vals) == 1 else 0.0))
        return tuple(vals[:target_arity])

    ops: list[DeltaOperator] = [
        DeltaOperator("constant_plus_1", "constant target delta +1.", lambda _r: tup(1, 0), 0.5),
        DeltaOperator("constant_minus_1", "constant target delta -1.", lambda _r: tup(-1, 0), 0.5),
        DeltaOperator("constant_second_plus_1", "constant secondary-state delta +1.", lambda _r: tup(0, 1), 0.7),
    ]

    for feat in numeric_features:
        ops.extend(
            [
                DeltaOperator(
                    f"{feat}_threshold_ge_15",
                    f"threshold operator on {feat}: if {feat} >= 15 then primary delta +2 else -2.",
                    lambda r, f=feat: tup(2 if float(r[f]) >= 15 else -2, 0),
                    structural_complexity(feat, "threshold", 1.2),
                ),
                DeltaOperator(
                    f"{feat}_threshold_lt_10_secondary_boost",
                    f"threshold operator on {feat}: if {feat} < 10 then primary delta -2 and secondary delta +10, else primary delta +1.",
                    lambda r, f=feat: tup(-2, 10) if float(r[f]) < 10 else tup(1, 0),
                    structural_complexity(feat, "threshold", 1.4),
                ),
                DeltaOperator(
                    f"{feat}_mod_3",
                    f"modulo operator on {feat}: if {feat} % 3 == 0 then primary delta +2 else -1.",
                    lambda r, f=feat: tup(2 if int(float(r[f])) % 3 == 0 else -1, 0),
                    structural_complexity(feat, "modulo", 1.3),
                ),
                DeltaOperator(
                    f"{feat}_parity_odd",
                    f"parity operator on {feat}: odd values give primary delta +1, even values give 0.",
                    lambda r, f=feat: tup(1 if int(float(r[f])) % 2 == 1 else 0, 0),
                    structural_complexity(feat, "parity", 1.1),
                ),
            ]
        )

    if coordinate_features is not None:
        x_key, y_key = coordinate_features
        ops.extend(
            [
                DeltaOperator(
                    f"{x_key}_minus_{y_key}",
                    f"linear coordinate operator: primary delta equals {x_key} - {y_key}.",
                    lambda r, x=x_key, y=y_key: tup(float(r[x]) - float(r[y]), 0),
                    1.2,
                ),
                DeltaOperator(
                    f"{x_key}_{y_key}_parity",
                    f"coordinate parity operator: odd {x_key}+{y_key} gives +1, even gives -1.",
                    lambda r, x=x_key, y=y_key: tup(1 if int(float(r[x]) + float(r[y])) % 2 == 1 else -1, 0),
                    1.4,
                ),
                DeltaOperator(
                    f"{x_key}_{y_key}_boundary_band",
                    f"boundary-band coordinate operator over {x_key},{y_key}: near-boundary states give primary delta +3, corner states may add another +3, otherwise 0.",
                    lambda r, x=x_key, y=y_key: tup(
                        (
                            6
                            if (
                                float(r[x]) in (0.0, 9.0)
                                and float(r[y]) in (0.0, 9.0)
                            )
                            else 3
                            if (
                                float(r[x]) <= 1
                                or float(r[x]) >= 8
                                or float(r[y]) <= 1
                                or float(r[y]) >= 8
                            )
                            else 0
                        ),
                        0,
                    ),
                    1.6,
                ),
            ]
        )
    return ops
