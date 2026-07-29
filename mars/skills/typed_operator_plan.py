"""Compile typed residual plans into safe, reusable executable layers.

The language model selects a small formal operator family.  This compiler,
rather than the model, instantiates that family against an observed interface.
The resulting layer never contains a source-task identifier, question, or
answer: field names and constants are discovered only when the layer executes
on its current context.
"""

from __future__ import annotations

import json
from typing import Any


_FAMILIES = {
    "table_numeric_association",
    "positive_monomial_lattice",
    "positive_additive_lattice",
    "common_string_suffix",
}


def admissible_families(*, input_type: str, target_type: str, signature_hint: str) -> list[str]:
    """Return interface-compatible families without inspecting task identity."""

    signature = signature_hint.lower()
    if input_type == "DataFrame" and "analyze(df)" in signature:
        return ["table_numeric_association"]
    if input_type == "dict" and target_type in {"float", "int"} and "inputs: dict" in signature:
        return ["positive_monomial_lattice", "positive_additive_lattice"]
    if input_type == "str" and target_type == "str" and "current: str" in signature:
        return ["common_string_suffix"]
    return []


def _table_association_layer() -> str:
    return r'''def build_layer(context: dict) -> dict:
    if "analyze(df)" not in str(context.get("signature_hint", "")):
        return {"program_sources": [], "signals": [], "residual_hints": []}
    code = """def analyze(df):
    cols = list(df.select_dtypes(include='number').columns)
    if len(cols) < 2:
        return {'evidence': 'No numeric variable pair is available for an association probe.', 'statistic': 0.0}
    corr = df[cols].corr(numeric_only=True)
    best = None
    best_abs = -1.0
    for i, left in enumerate(cols):
        for right in cols[i + 1:]:
            value = corr.loc[left, right]
            if value == value and abs(float(value)) > best_abs:
                best_abs = abs(float(value))
                best = (left, right, float(value))
    if best is None:
        return {'evidence': 'No finite numeric association was found.', 'statistic': 0.0}
    direction = 'positive' if best[2] >= 0 else 'negative'
    return {'evidence': 'strongest numeric association: corr(' + str(best[0]) + ',' + str(best[1]) + ')=' + str(round(best[2], 4)) + ' (' + direction + ')', 'statistic': abs(best[2])}
"""
    return {
        "program_sources": [{
            "name": "typed_numeric_association",
            "description": "measure the strongest dynamically discovered numeric association",
            "complexity": 1.4,
            "code": code,
        }],
        "operator_sources": [],
        "signals": ["typed plan: dynamic numeric-pair association"],
        "residual_hints": ["numeric association probe added from a recurring residual class"],
        "metadata": {"family": "table_numeric_association"},
    }
'''


def _monomial_lattice_layer() -> str:
    return r'''def build_layer(context: dict) -> dict:
    if "inputs: dict" not in str(context.get("signature_hint", "")):
        return {"program_sources": [], "signals": [], "residual_hints": []}
    rows = list(context.get("observations_typed") or context.get("observations") or [])
    if len(rows) < 3:
        return {"program_sources": [], "signals": [], "residual_hints": ["need at least three numeric observations"]}
    first = rows[0].get("inputs") if isinstance(rows[0], dict) else None
    if not isinstance(first, dict):
        return {"program_sources": [], "signals": [], "residual_hints": []}
    keys = []
    for key in sorted(first):
        values = []
        valid = True
        for row in rows:
            inputs = row.get("inputs") if isinstance(row, dict) else None
            try:
                value = float(inputs[key])
            except Exception:
                valid = False
                break
            if value != value or value == float("inf") or value == float("-inf") or value <= 0:
                valid = False
                break
            values.append(value)
        if valid:
            keys.append(key)
    targets = []
    for row in rows:
        try:
            value = float(row.get("target"))
        except Exception:
            return {"program_sources": [], "signals": [], "residual_hints": []}
        if value != value or value == float("inf") or value == float("-inf") or value <= 0:
            return {"program_sources": [], "signals": [], "residual_hints": []}
        targets.append(value)
    keys = keys[:4]
    if not keys:
        return {"program_sources": [], "signals": [], "residual_hints": ["positive scalar interface has no stable input keys"]}
    candidates = []
    def score_powers(powers):
        if not any(powers):
            return
        ratios = []
        for row, target in zip(rows, targets):
            base = 1.0
            for key, power in zip(keys, powers):
                base = base * (float(row["inputs"][key]) ** power)
            if base == 0 or base != base:
                return
            ratios.append(target / base)
        ratios.sort()
        center = ratios[len(ratios) // 2]
        if center == 0 or center != center:
            return
        error = sum(abs((value - center) / center) for value in ratios) / len(ratios)
        candidates.append((error + 0.015 * sum(abs(p) for p in powers), tuple(powers)))
    def enumerate_powers(prefix):
        if len(prefix) == len(keys):
            score_powers(prefix)
            return
        for power in (-2, -1, 0, 1, 2):
            enumerate_powers(prefix + [power])
    enumerate_powers([])
    candidates.sort(key=lambda item: item[0])
    programs = []
    for rank, (_, powers) in enumerate(candidates[:4]):
        terms = []
        for key, power in zip(keys, powers):
            if power:
                escaped = str(key).replace("\\", "\\\\").replace("'", "\\'")
                terms.append("(inputs['" + escaped + "'] ** " + str(power) + ")")
        expression = " * ".join(terms) if terms else "1.0"
        programs.append({
            "name": "typed_monomial_lattice_" + str(rank),
            "description": "positive-input monomial selected by a residual-conditioned exponent lattice",
            "complexity": 1.0 + 0.2 * sum(abs(p) for p in powers),
            "code": "def law(inputs: dict) -> float:\n    return " + expression + "\n",
        })
    return {
        "program_sources": programs,
        "operator_sources": [],
        "signals": ["typed plan: positive monomial exponent lattice over observed input keys"],
        "residual_hints": ["power-law coordinate probe added from a recurring numeric residual class"],
        "metadata": {"family": "positive_monomial_lattice", "n_keys": len(keys)},
    }
'''


def _additive_lattice_layer() -> str:
    return r'''def build_layer(context: dict) -> dict:
    if "inputs: dict" not in str(context.get("signature_hint", "")):
        return {"program_sources": [], "signals": [], "residual_hints": []}
    rows = list(context.get("observations_typed") or context.get("observations") or [])
    if len(rows) < 3:
        return {"program_sources": [], "signals": [], "residual_hints": ["need at least three numeric observations"]}
    first = rows[0].get("inputs") if isinstance(rows[0], dict) else None
    if not isinstance(first, dict):
        return {"program_sources": [], "signals": [], "residual_hints": []}
    keys = []
    for key in sorted(first):
        try:
            values = [float(row["inputs"][key]) for row in rows]
        except Exception:
            continue
        if all(value == value and value not in (float("inf"), float("-inf")) for value in values):
            keys.append(key)
    keys = keys[:4]
    if not keys:
        return {"program_sources": [], "signals": [], "residual_hints": ["numeric interface has no stable input keys"]}
    programs = []
    for rank, key in enumerate(keys):
        escaped = str(key).replace("\\", "\\\\").replace("'", "\\'")
        programs.append({
            "name": "typed_additive_coordinate_" + str(rank),
            "description": "single-coordinate additive probe selected from the observed numeric interface",
            "complexity": 1.2,
            "code": "def law(inputs: dict) -> float:\n    return inputs['" + escaped + "']\n",
        })
    if len(keys) >= 2:
        left = str(keys[0]).replace("\\", "\\\\").replace("'", "\\'")
        right = str(keys[1]).replace("\\", "\\\\").replace("'", "\\'")
        programs.append({
            "name": "typed_additive_pair",
            "description": "two-coordinate additive probe selected from the observed numeric interface",
            "complexity": 1.4,
            "code": "def law(inputs: dict) -> float:\n    return inputs['" + left + "'] + inputs['" + right + "']\n",
        })
    return {
        "program_sources": programs,
        "operator_sources": [],
        "signals": ["typed plan: additive coordinate lattice over observed input keys"],
        "residual_hints": ["additive coordinate probe added from a recurring numeric residual class"],
        "metadata": {"family": "positive_additive_lattice", "n_keys": len(keys)},
    }
'''


def _common_suffix_layer() -> str:
    return r'''def build_layer(context: dict) -> dict:
    if "current: str" not in str(context.get("signature_hint", "")):
        return {"program_sources": [], "signals": [], "residual_hints": []}
    rows = list(context.get("observations_typed") or context.get("observations") or [])
    suffixes = []
    for row in rows:
        current = str(row.get("inputs", ""))
        target = str(row.get("target", ""))
        if not target.startswith(current):
            return {"program_sources": [], "signals": [], "residual_hints": ["prefix residual class did not recur"]}
        suffixes.append(target[len(current):])
    if not suffixes or any(item != suffixes[0] for item in suffixes):
        return {"program_sources": [], "signals": [], "residual_hints": ["no common suffix in residual class"]}
    literal = repr(suffixes[0])
    return {
        "program_sources": [{
            "name": "typed_common_suffix",
            "description": "append the suffix shared across residuals",
            "complexity": 1.1,
            "code": "def rule(current: str, context: dict) -> str:\n    return current + " + literal + "\n",
        }],
        "operator_sources": [],
        "signals": ["typed plan: common string suffix"],
        "residual_hints": ["shared suffix was compiled from observed residuals"],
        "metadata": {"family": "common_string_suffix"},
    }
'''


_LAYER_BUILDERS = {
    "table_numeric_association": _table_association_layer,
    "positive_monomial_lattice": _monomial_lattice_layer,
    "positive_additive_lattice": _additive_lattice_layer,
    "common_string_suffix": _common_suffix_layer,
}


def compile_typed_operator_plan(
    plan: dict[str, Any],
    *,
    input_type: str,
    target_type: str,
    signature_hint: str,
) -> dict[str, Any] | None:
    """Compile a type-valid plan into a layer specification or reject it."""

    family = str(plan.get("family", ""))
    if family not in _FAMILIES:
        return None
    if family not in admissible_families(
        input_type=input_type,
        target_type=target_type,
        signature_hint=signature_hint,
    ):
        return None
    return {
        "name": f"typed_{family}",
        "layer_type": "typed_residual_operator",
        "description": str(plan.get("description") or f"typed residual plan: {family}"),
        "contract": {
            "input": "context: dict",
            "output": "dict(program_sources, operator_sources, signals, residual_hints, metadata)",
            "scope": "typed_residual_class",
            "family": family,
        },
        "code": _LAYER_BUILDERS[family]().strip(),
        "plan": json.loads(json.dumps(plan)),
    }
