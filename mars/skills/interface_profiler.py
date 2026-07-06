"""Benchmark-agnostic interface profiling for weak-model assistance.

The profiler compresses raw observations into a small typed map: input family,
target family, candidate axes, candidate variables, and likely operator families.
It does not know benchmark names or hidden answers.  It is meant to reduce the
reasoning burden on small models before they write executable hypotheses.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class InterfaceProfile:
    input_family: str
    target_family: str
    n_observations: int
    context_keys: tuple[str, ...]
    input_features: tuple[str, ...]
    target_features: tuple[str, ...]
    candidate_axes: tuple[str, ...]
    candidate_variables: tuple[str, ...]
    operator_families: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_prompt(self) -> str:
        return (
            "\nUNIVERSAL INTERFACE PROFILE (data-derived, not an answer):\n"
            + _jsonish(self.to_dict())
            + "\nUse this profile to decompose the task into typed holes. "
            "Treat candidate_axes as indexing/context variables, not target "
            "scientific variables unless the task explicitly asks for the axis."
        )


def profile_observations(observations: Iterable[Any]) -> InterfaceProfile:
    obs = list(observations)
    if not obs:
        return InterfaceProfile(
            input_family="empty",
            target_family="unknown",
            n_observations=0,
            context_keys=(),
            input_features=(),
            target_features=(),
            candidate_axes=(),
            candidate_variables=(),
            operator_families=("collect_more_observations",),
            warnings=("no_observations",),
        )

    inputs = [getattr(o, "inputs", None) for o in obs]
    targets = [getattr(o, "target", None) for o in obs]
    contexts = [getattr(o, "context", {}) or {} for o in obs]
    input_family = _family(inputs[0])
    target_family = _family(targets[0])
    context_keys = tuple(sorted({str(k) for ctx in contexts for k in ctx if not str(k).startswith("__")})[:32])

    input_features, axes, variables = _input_profile(inputs)
    target_features = _target_profile(targets)
    operator_families = _operator_families(
        input_family=input_family,
        target_family=target_family,
        input_features=input_features,
        target_features=target_features,
        axes=axes,
        variables=variables,
        contexts=contexts,
    )
    warnings = _warnings(input_family, target_family, axes, variables, input_features)
    return InterfaceProfile(
        input_family=input_family,
        target_family=target_family,
        n_observations=len(obs),
        context_keys=context_keys,
        input_features=input_features,
        target_features=target_features,
        candidate_axes=axes,
        candidate_variables=variables,
        operator_families=operator_families,
        warnings=warnings,
    )


def profile_prompt(observations: Iterable[Any]) -> str:
    return profile_observations(observations).to_prompt()


def _input_profile(inputs: list[Any]) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    first = inputs[0]
    if _is_dataframe(first):
        return _dataframe_profile(first)
    if isinstance(first, Mapping):
        keys = tuple(str(k) for item in inputs for k in getattr(item, "keys", lambda: [])())
        uniq = tuple(dict.fromkeys(keys))
        axes = tuple(k for k in uniq if _is_axis_name(k.lower()))[:12]
        variables = tuple(k for k in uniq if k not in axes)[:24]
        features = [f"mapping_keys={len(uniq)}"]
        numeric = []
        for key in uniq[:16]:
            vals = [item.get(key) for item in inputs if isinstance(item, Mapping)]
            if _numeric_fraction(vals) >= 0.8:
                numeric.append(key)
        if numeric:
            features.append("numeric_keys=" + ",".join(numeric[:12]))
        return tuple(features), axes, variables
    if isinstance(first, str):
        lens = [len(str(x)) for x in inputs]
        alphabet = sorted({ch for x in inputs for ch in str(x)})[:16]
        features = (
            f"string_len_min={min(lens)}",
            f"string_len_max={max(lens)}",
            "alphabet=" + "".join(alphabet),
        )
        return features, (), ("sequence",)
    if isinstance(first, (list, tuple)):
        lens = [len(x) for x in inputs if isinstance(x, (list, tuple))]
        features = (f"sequence_len_min={min(lens)}", f"sequence_len_max={max(lens)}") if lens else ("sequence",)
        return features, (), ("sequence",)
    return (f"type={type(first).__name__}",), (), ()


def _dataframe_profile(df: Any) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    cols = [str(c) for c in getattr(df, "columns", [])]
    axes: list[str] = []
    variables: list[str] = []
    numeric_cols: list[str] = []
    text_cols: list[str] = []
    for col in cols:
        low = col.lower()
        if _is_axis_name(low):
            axes.append(col)
            continue
        series = getattr(df, "__getitem__", lambda _c: [])(col)
        frac = _numeric_fraction(list(series)[:200] if hasattr(series, "__iter__") else [])
        if frac >= 0.7:
            numeric_cols.append(col)
            variables.append(col)
        else:
            text_cols.append(col)
    features = [
        f"table_rows={len(df) if hasattr(df, '__len__') else 'unknown'}",
        f"table_cols={len(cols)}",
    ]
    if numeric_cols:
        features.append("numeric_cols=" + ",".join(numeric_cols[:12]))
    if text_cols:
        features.append("text_cols=" + ",".join(text_cols[:8]))
    return tuple(features), tuple(axes[:16]), tuple(variables[:32])


def _target_profile(targets: list[Any]) -> tuple[str, ...]:
    first = targets[0]
    family = _family(first)
    if family == "number":
        vals = [float(x) for x in targets if _is_number(x)]
        if not vals:
            return ("numeric_target",)
        return (
            f"target_min={min(vals):.4g}",
            f"target_max={max(vals):.4g}",
            f"target_unique={len(set(round(v, 8) for v in vals))}",
        )
    if family == "string":
        lens = [len(str(x)) for x in targets]
        return (f"target_string_len_min={min(lens)}", f"target_string_len_max={max(lens)}")
    if family == "mapping":
        keys = tuple(sorted({str(k) for t in targets if isinstance(t, Mapping) for k in t.keys()}))
        return ("target_keys=" + ",".join(keys[:16]),)
    return (f"target_family={family}",)


def _operator_families(
    *,
    input_family: str,
    target_family: str,
    input_features: tuple[str, ...],
    target_features: tuple[str, ...],
    axes: tuple[str, ...],
    variables: tuple[str, ...],
    contexts: list[dict[str, Any]],
) -> tuple[str, ...]:
    ops: list[str] = []
    if input_family == "table":
        if axes and variables:
            ops.extend([
                "axis_variable_separation",
                "single_series_event_probe",
                "multi_series_relation_probe",
                "windowed_change_probe",
            ])
        if any("text_cols=" in f for f in input_features):
            ops.append("wide_table_row_entity_probe")
        if len(variables) >= 2:
            ops.append("association_or_comparison_probe")
    if input_family == "mapping" and target_family == "number":
        ops.extend(["numeric_formula_search", "power_law_probe", "constant_calibration"])
    if input_family == "string" and target_family == "string":
        ops.extend(["string_transform_probe", "positionwise_rule_probe", "context_branch_probe"])
    if any("step" in str(k).lower() for ctx in contexts for k in ctx):
        ops.append("step_condition_branch_probe")
    if target_family == "mapping":
        ops.append("schema_completion_probe")
    if not ops:
        ops.append("typed_residual_probe")
    return tuple(dict.fromkeys(ops))


def _warnings(
    input_family: str,
    target_family: str,
    axes: tuple[str, ...],
    variables: tuple[str, ...],
    input_features: tuple[str, ...],
) -> tuple[str, ...]:
    warnings: list[str] = []
    if input_family == "table" and axes:
        warnings.append("axis_columns_are_context_not_scientific_variables")
    if input_family == "table" and not variables:
        warnings.append("no_numeric_candidate_variables_detected")
    if target_family == "none":
        warnings.append("no_direct_target_use_self_consistency_or_report_grounding")
    if any("table_cols=" in f for f in input_features):
        try:
            n_cols = int(next(f.split("=", 1)[1] for f in input_features if f.startswith("table_cols=")))
            if n_cols > 40:
                warnings.append("wide_table_needs_entity_or_row_selection_before_correlation")
        except Exception:
            pass
    return tuple(warnings)


def _family(value: Any) -> str:
    if value is None:
        return "none"
    if _is_dataframe(value):
        return "table"
    if isinstance(value, Mapping):
        return "mapping"
    if isinstance(value, str):
        return "string"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return "number"
    if isinstance(value, (list, tuple)):
        return "sequence"
    return type(value).__name__


def _is_dataframe(value: Any) -> bool:
    return hasattr(value, "columns") and hasattr(value, "__getitem__") and hasattr(value, "shape")


def _is_axis_name(low: str) -> bool:
    return (
        low in {"year", "bce", "ce", "calbp", "time", "date", "id", "index"}
        or low.endswith("_id")
        or low.startswith("id_")
        or re.match(r"^\d{4}(?:\b|\s|\[|_)", low) is not None
        or re.match(r"^yr\d{4}\b", low) is not None
        or re.match(r"^\d{4}\s*\[yr\d{4}\]", low) is not None
    )


def _numeric_fraction(values: list[Any]) -> float:
    if not values:
        return 0.0
    good = sum(1 for v in values if _is_number(v))
    return good / len(values)


def _is_number(value: Any) -> bool:
    try:
        return math.isfinite(float(str(value).replace(",", ".")))
    except Exception:
        return False


def _jsonish(obj: Any) -> str:
    import json

    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)[:3000]
