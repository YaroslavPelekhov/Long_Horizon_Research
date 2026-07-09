"""Typed failure algebra for Darwinian theory synthesis.

The point of this module is to avoid asking a weak model to invent a whole
hypothesis.  We first turn observable failures and interface structure into
typed missing-gene queries.  A synthesizer can then search over small formal
edits instead of unconstrained natural-language ideas.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class FailureSyndrome:
    """A typed, benchmark-agnostic pattern that suggests a missing gene."""

    kind: str
    severity: float
    evidence: str
    bits: Mapping[str, Any]


@dataclass(frozen=True)
class MissingGeneQuery:
    """A typed request for a small theory-genome edit."""

    gene_type: str
    reason: str
    constraints: Mapping[str, Any]
    source_syndrome: str


def infer_failure_syndromes(
    *,
    signature_hint: str,
    observations: Sequence[Any],
    question: str = "",
    column_descriptions: Mapping[str, str] | None = None,
    domain_context: str = "",
) -> tuple[FailureSyndrome, ...]:
    """Infer generic failure/pressure types from the visible task interface."""

    if not observations:
        return ()
    hint = str(signature_hint)
    syndromes: list[FailureSyndrome] = []
    if "analyze(df)" in hint:
        syndromes.extend(
            _table_syndromes(
                observations,
                question=question,
                column_descriptions=column_descriptions or {},
                domain_context=domain_context,
            )
        )
    if "law(inputs" in hint:
        numeric = _numeric_law_syndrome(observations)
        if numeric is not None:
            syndromes.append(numeric)
    if "rule(current" in hint:
        transition = _string_transition_syndrome(observations)
        if transition is not None:
            syndromes.append(transition)
    if "predict(parent1" in hint:
        inherited = _dict_transition_syndrome(observations)
        if inherited is not None:
            syndromes.append(inherited)
    return tuple(syndromes)


def missing_gene_queries(
    syndromes: Sequence[FailureSyndrome],
) -> tuple[MissingGeneQuery, ...]:
    """Map typed syndromes to typed missing-gene queries."""

    queries: list[MissingGeneQuery] = []
    for syn in syndromes:
        if syn.kind == "wide_table_axis_ambiguity":
            queries.append(
                MissingGeneQuery(
                    gene_type="axis_orientation_gene",
                    reason="many axis-like columns and semantic row labels imply variables may live in rows",
                    constraints=dict(syn.bits),
                    source_syndrome=syn.kind,
                )
            )
        elif syn.kind == "semantic_role_binding_pressure":
            queries.append(
                MissingGeneQuery(
                    gene_type="role_binding_gene",
                    reason="question tokens must be bound to observable objects before relations are tested",
                    constraints=dict(syn.bits),
                    source_syndrome=syn.kind,
                )
            )
        elif syn.kind == "numeric_scalar_law_pressure":
            queries.append(
                MissingGeneQuery(
                    gene_type="scale_relation_gene",
                    reason="numeric dict inputs and scalar targets support a compact executable law",
                    constraints=dict(syn.bits),
                    source_syndrome=syn.kind,
                )
            )
        elif syn.kind == "string_transition_pressure":
            queries.append(
                MissingGeneQuery(
                    gene_type="transition_rule_gene",
                    reason="string inputs and targets require an executable transition theory",
                    constraints=dict(syn.bits),
                    source_syndrome=syn.kind,
                )
            )
        elif syn.kind == "dict_transition_pressure":
            queries.append(
                MissingGeneQuery(
                    gene_type="state_role_gene",
                    reason="structured parent/state inputs require role-level state aggregation",
                    constraints=dict(syn.bits),
                    source_syndrome=syn.kind,
                )
            )
    return tuple(queries)


def _table_syndromes(
    observations: Sequence[Any],
    *,
    question: str,
    column_descriptions: Mapping[str, str],
    domain_context: str,
) -> list[FailureSyndrome]:
    df = getattr(observations[-1], "inputs", None)
    if df is None or not hasattr(df, "columns"):
        return []
    columns = [str(c) for c in getattr(df, "columns", [])]
    year_cols = [c for c in columns if _is_year_like(c)]
    text_cols = [c for c in columns if c not in year_cols]
    syndromes: list[FailureSyndrome] = []
    if len(year_cols) >= 5 and text_cols:
        syndromes.append(
            FailureSyndrome(
                kind="wide_table_axis_ambiguity",
                severity=min(1.0, len(year_cols) / max(8.0, len(columns))),
                evidence=(
                    f"detected {len(year_cols)} year-like columns and "
                    f"{len(text_cols)} non-axis label candidates"
                ),
                bits={
                    "axis_candidates": year_cols[:64],
                    "label_candidates": text_cols[:12],
                    "n_columns": len(columns),
                },
            )
        )
    q_tokens = _tokens(f"{question} {domain_context}")
    schema_text = " ".join(f"{c} {column_descriptions.get(c, '')}" for c in columns)
    schema_tokens = _tokens(schema_text)
    if q_tokens and (q_tokens & schema_tokens or text_cols):
        syndromes.append(
            FailureSyndrome(
                kind="semantic_role_binding_pressure",
                severity=0.7,
                evidence="question/schema tokens must be bound into typed roles before relation search",
                bits={
                    "question_tokens": sorted(q_tokens)[:48],
                    "schema_tokens": sorted(schema_tokens)[:48],
                    "object_candidates": columns[:64],
                },
            )
        )
    return syndromes


def _numeric_law_syndrome(observations: Sequence[Any]) -> FailureSyndrome | None:
    vars_seen: list[str] = []
    n_numeric = 0
    for obs in observations:
        inputs = getattr(obs, "inputs", None)
        target = getattr(obs, "target", None)
        if not isinstance(inputs, dict):
            continue
        try:
            float(target)
        except Exception:
            continue
        numeric_keys = []
        for key, value in inputs.items():
            try:
                float(value)
            except Exception:
                continue
            numeric_keys.append(str(key))
        if numeric_keys:
            n_numeric += 1
            for key in numeric_keys:
                if key not in vars_seen:
                    vars_seen.append(key)
    if n_numeric < 2 or not vars_seen:
        return None
    return FailureSyndrome(
        kind="numeric_scalar_law_pressure",
        severity=min(1.0, n_numeric / max(3.0, len(observations))),
        evidence=f"detected scalar numeric target with {len(vars_seen)} numeric input variables",
        bits={"variables": vars_seen[:16], "n_numeric_observations": n_numeric},
    )


def _string_transition_syndrome(observations: Sequence[Any]) -> FailureSyndrome | None:
    pairs = []
    context_keys: list[str] = []
    for obs in observations:
        inputs = getattr(obs, "inputs", None)
        target = getattr(obs, "target", None)
        if not isinstance(inputs, str) or not isinstance(target, str):
            continue
        pairs.append((inputs, target))
        for key in getattr(obs, "context", {}) or {}:
            if key not in context_keys:
                context_keys.append(str(key))
    if len(pairs) < 2:
        return None
    return FailureSyndrome(
        kind="string_transition_pressure",
        severity=min(1.0, len(pairs) / max(3.0, len(observations))),
        evidence=f"detected {len(pairs)} string input-target transition examples",
        bits={"context_keys": context_keys[:24], "n_pairs": len(pairs)},
    )


def _dict_transition_syndrome(observations: Sequence[Any]) -> FailureSyndrome | None:
    input_keys: list[str] = []
    target_keys: list[str] = []
    n_structured = 0
    for obs in observations:
        inputs = getattr(obs, "inputs", None)
        target = getattr(obs, "target", None)
        if not isinstance(inputs, dict) or not isinstance(target, dict):
            continue
        n_structured += 1
        for key in inputs:
            if str(key) not in input_keys:
                input_keys.append(str(key))
        for key in target:
            if str(key) not in target_keys:
                target_keys.append(str(key))
    if n_structured < 2:
        return None
    return FailureSyndrome(
        kind="dict_transition_pressure",
        severity=min(1.0, n_structured / max(3.0, len(observations))),
        evidence=f"detected {n_structured} structured transition examples",
        bits={"input_keys": input_keys[:24], "target_keys": target_keys[:24]},
    )


def _tokens(text: str) -> set[str]:
    stop = {
        "the", "and", "for", "with", "from", "into", "between", "among", "which",
        "what", "when", "where", "does", "did", "were", "was", "are", "how",
        "this", "that", "have", "has", "had", "over", "under", "after", "before",
        "data", "table", "variable", "variables", "column", "columns", "row", "rows",
    }
    return {
        tok
        for tok in re.findall(r"[a-zA-Z][a-zA-Z0-9_/-]{2,}", str(text).lower())
        if tok not in stop
    }


def _is_year_like(value: str) -> bool:
    return bool(re.search(r"(18|19|20|21)\d{2}", str(value)))
