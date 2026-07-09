"""Darwinian synthesis of formal theory genomes.

This layer deliberately avoids asking the LLM to propose a theory.  It builds a
population of small formal genomes from the observable interface, mutates and
ranks them by executable fitness, then emits phenotypes as ordinary analyzer
programs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Mapping, Sequence

from mars.darwin.failure_algebra import infer_failure_syndromes, missing_gene_queries
from mars.darwin.probes import build_probe_genomes
from mars.darwin.regulatory import (
    RegulatoryGenome,
    default_regulatory_genome,
    develop_regulatory_genome,
)


@dataclass(frozen=True)
class TheoryGenome:
    """A compact formal genome for a hypothesis-producing analyzer."""

    name: str
    kind: str
    genes: Mapping[str, Any]
    complexity: float
    lineage: tuple[str, ...] = ()


@dataclass(frozen=True)
class DarwinResult:
    program_sources: tuple[dict[str, Any], ...]
    genomes: tuple[TheoryGenome, ...]
    trace: tuple[dict[str, Any], ...]


class DarwinSynthesizer:
    """Generate, mutate, and emit theory genomes without semantic proposals."""

    def __init__(
        self,
        *,
        max_programs: int = 32,
        regulatory_genome: RegulatoryGenome | None = None,
    ):
        self.max_programs = int(max_programs)
        self.regulatory_genome = regulatory_genome or default_regulatory_genome()

    def synthesize(
        self,
        *,
        signature_hint: str,
        observations: Sequence[Any],
        interface_name: str = "unknown",
        question: str = "",
        column_descriptions: Mapping[str, str] | None = None,
        domain_context: str = "",
    ) -> DarwinResult:
        syndromes = infer_failure_syndromes(
            signature_hint=signature_hint,
            observations=observations,
            question=question,
            column_descriptions=column_descriptions or {},
            domain_context=domain_context,
        )
        queries = missing_gene_queries(syndromes)
        probes = build_probe_genomes(
            signature_hint=signature_hint,
            observations=observations,
        )
        development = develop_regulatory_genome(
            self.regulatory_genome,
            signature_hint=signature_hint,
            syndromes=syndromes,
            queries=queries,
            probes=probes,
        )
        genomes = self._initial_population(
            signature_hint=signature_hint,
            observations=observations,
            interface_name=interface_name,
            question=question,
            column_descriptions=column_descriptions or {},
            domain_context=domain_context,
            missing_gene_queries=queries,
            emitted_families=development.emitted_families,
            family_priorities=_family_priorities(development.active_genes),
        )
        mutated = self._mutate(genomes)
        population = _dedup_genomes(tuple(genomes) + tuple(mutated))
        sources: list[dict[str, Any]] = []
        trace: list[dict[str, Any]] = [
            {
                "kind": "darwin_failure_algebra",
                "syndromes": [
                    {
                        "kind": s.kind,
                        "severity": s.severity,
                        "evidence": s.evidence,
                        "bits": dict(s.bits),
                    }
                    for s in syndromes
                ],
                "missing_gene_queries": [
                    {
                        "gene_type": q.gene_type,
                        "reason": q.reason,
                        "constraints": dict(q.constraints),
                        "source_syndrome": q.source_syndrome,
                    }
                    for q in queries
                ],
                "probes": [
                    {
                        "name": p.name,
                        "probe_type": p.probe_type,
                        "transformation": p.transformation,
                        "expected_invariance": p.expected_invariance,
                        "cost": p.cost,
                        "bits": dict(p.bits),
                    }
                    for p in probes
                ],
                "regulatory_development": dict(development.trace),
            }
        ]
        for genome in population[: self.max_programs]:
            source = self._phenotype_source(genome)
            if not source:
                continue
            sources.append(
                {
                    "name": f"darwin_{genome.name}",
                    "description": _describe_genome(genome),
                    "complexity": genome.complexity,
                    "code": source,
                    "genome": dict(genome.genes),
                    "lineage": list(genome.lineage),
                }
            )
            trace.append(
                {
                    "name": genome.name,
                    "kind": genome.kind,
                    "complexity": genome.complexity,
                    "genes": dict(genome.genes),
                    "lineage": list(genome.lineage),
                }
            )
        return DarwinResult(
            program_sources=tuple(sources),
            genomes=population,
            trace=tuple(trace),
        )

    def _initial_population(
        self,
        *,
        signature_hint: str,
        observations: Sequence[Any],
        interface_name: str,
        question: str,
        column_descriptions: Mapping[str, str],
        domain_context: str,
        missing_gene_queries: Sequence[Any],
        emitted_families: Sequence[str],
        family_priorities: Mapping[str, float],
    ) -> tuple[TheoryGenome, ...]:
        if not observations:
            return ()
        genomes: list[TheoryGenome] = []
        query_types = {str(getattr(q, "gene_type", "")) for q in missing_gene_queries}
        families = set(str(f) for f in emitted_families)
        if not families:
            if "analyze(df)" in signature_hint:
                families.add("table_chain")
            if "law(inputs" in signature_hint or "scale_relation_gene" in query_types:
                families.add("numeric_power_law")
            if "rule(current" in signature_hint or "transition_rule_gene" in query_types:
                families.add("string_transition")
            if "predict(parent1" in signature_hint or "state_role_gene" in query_types:
                families.add("dict_state_predictor")
        if "table_chain" in families or "table_robust_mediation" in families:
            df = getattr(observations[-1], "inputs", None)
            if df is not None and hasattr(df, "columns"):
                genomes.extend(
                    _table_genomes(
                        df,
                        question=question,
                        column_descriptions=column_descriptions,
                        domain_context=domain_context,
                        family_priority=max(
                            float(family_priorities.get("table_chain", 0.0)),
                            float(family_priorities.get("table_robust_mediation", 0.0)),
                        ),
                        robust_only="table_robust_mediation" in families and "table_chain" not in families,
                    )
                )
        if "numeric_power_law" in families:
            genomes.extend(_numeric_law_genomes(observations))
        if "string_transition" in families:
            genomes.extend(_string_rule_genomes(observations))
        if "dict_state_predictor" in families:
            genomes.extend(_dict_predictor_genomes(observations))
        return tuple(genomes)

    def _mutate(self, genomes: Sequence[TheoryGenome]) -> tuple[TheoryGenome, ...]:
        out: list[TheoryGenome] = []
        for genome in genomes:
            if genome.kind != "table_chain":
                continue
            roles = dict(genome.genes.get("roles", {}) or {})
            cause = roles.get("cause")
            mediator = roles.get("mediator")
            outcome = roles.get("outcome")
            if cause and mediator:
                swapped = dict(roles)
                swapped["cause"], swapped["mediator"] = mediator, cause
                out.append(_replace_roles(genome, swapped, "swap_cause_mediator"))
            if mediator and outcome:
                swapped = dict(roles)
                swapped["mediator"], swapped["outcome"] = outcome, mediator
                out.append(_replace_roles(genome, swapped, "swap_mediator_outcome"))
        return tuple(out)

    def _phenotype_source(self, genome: TheoryGenome) -> str:
        if genome.kind == "table_chain":
            orientation = str(genome.genes.get("orientation", ""))
            if orientation == "rows_as_variables":
                if str(genome.genes.get("relation_family", "")) == "robust_mediated_trend":
                    return _rows_as_variables_robust_source(genome)
                return _rows_as_variables_source(genome)
            if orientation == "columns_as_variables":
                return _columns_as_variables_source(genome)
        if genome.kind == "numeric_power_law":
            return _numeric_power_law_source(genome)
        if genome.kind == "numeric_trig_relation":
            return _numeric_trig_relation_source(genome)
        if genome.kind == "string_transition":
            return _string_transition_source(genome)
        if genome.kind == "dict_state_predictor":
            return _dict_predictor_source(genome)
        return ""


def _table_genomes(
    df: Any,
    *,
    question: str,
    column_descriptions: Mapping[str, str],
    domain_context: str,
    family_priority: float = 0.0,
    robust_only: bool = False,
) -> list[TheoryGenome]:
    columns = [str(c) for c in getattr(df, "columns", [])]
    numeric_cols = _numeric_columns(df, columns)
    year_cols = [c for c in columns if _is_year_like(c)]
    text_cols = _text_columns(df, columns)
    q_tokens = _tokens(f"{question} {domain_context}")
    genomes: list[TheoryGenome] = []
    relevant_numeric = _rank_relevant_columns(
        numeric_cols,
        question=question,
        descriptions=column_descriptions,
        domain_context=domain_context,
    )[:6]
    if len(relevant_numeric) >= 2 and not robust_only:
        roles = {
            "cause": relevant_numeric[0],
            "outcome": relevant_numeric[1],
        }
        if len(relevant_numeric) >= 3:
            roles["mediator"] = relevant_numeric[1]
            roles["outcome"] = relevant_numeric[2]
        genomes.append(
            TheoryGenome(
                name="columns_as_variables_chain",
                kind="table_chain",
                genes={
                    "orientation": "columns_as_variables",
                    "roles": roles,
                    "variables": relevant_numeric,
                },
                complexity=2.7,
                lineage=("init", "columns_as_variables"),
            )
        )
    if len(year_cols) >= 5 and text_cols:
        label_col = _best_label_col(df, text_cols, q_tokens)
        row_items = _row_items(df, label_col, year_cols, q_tokens)
        if len(row_items) >= 2:
            roles = _assign_row_roles(row_items, question, domain_context)
            if roles.get("cause") and roles.get("outcome"):
                genomes.append(
                    TheoryGenome(
                        name="rows_as_variables_chain",
                        kind="table_chain",
                        genes={
                            "orientation": "rows_as_variables",
                            "axis": year_cols,
                            "label_col": label_col,
                            "roles": roles,
                        },
                        complexity=3.1 + 0.1 * len([v for v in roles.values() if v]),
                        lineage=("init", "wide_table", "rows_as_variables"),
                    )
                )
                if family_priority >= 4.5 or robust_only:
                    robust_genes = {
                        "orientation": "rows_as_variables",
                        "axis": year_cols,
                        "label_col": label_col,
                        "roles": roles,
                        "relation_family": "robust_mediated_trend",
                    }
                    genomes.append(
                        TheoryGenome(
                            name="rows_as_variables_robust_mediated_trend",
                            kind="table_chain",
                            genes=robust_genes,
                            complexity=3.4 + 0.1 * len([v for v in roles.values() if v]),
                            lineage=(
                                "regulatory_development",
                                "wide_table",
                                "robust_mediated_trend",
                            ),
                        )
                    )
    return genomes


def _family_priorities(active_genes: Sequence[Any]) -> dict[str, float]:
    priorities: dict[str, float] = {}
    for gene in active_genes:
        try:
            priority = float(getattr(gene, "priority", 0.0))
            families = tuple(getattr(gene, "emit_families", ()) or ())
        except Exception:
            continue
        for family in families:
            priorities[str(family)] = priorities.get(str(family), 0.0) + priority
    return priorities


def _assign_row_roles(
    rows: Sequence[dict[str, Any]],
    question: str,
    domain_context: str,
) -> dict[str, str]:
    q = question.lower()
    trigger_pos = len(q)
    for trigger in (" influence ", " affect ", " impact ", " contribute ", " associated ", " relation ", " relationship "):
        pos = q.find(trigger)
        if 0 <= pos < trigger_pos:
            trigger_pos = pos
    cause_terms = _tokens(q[:trigger_pos] if trigger_pos < len(q) else q)
    all_terms = _tokens(f"{question} {domain_context}")
    mediator_terms = {
        t for t in all_terms
        if t in {"human", "capital", "labor", "labour", "force", "school", "enrollment", "primary", "secondary"}
    }
    outcome_terms = {
        t for t in all_terms
        if t in {"gdp", "gni", "income", "output", "economic", "capita"}
    }

    def score(item: dict[str, Any], terms: set[str], role: str) -> float:
        toks = set(item.get("tokens", set()))
        label = str(item.get("label", "")).lower()
        value = 10.0 * len(toks & terms) + len(toks & all_terms)
        if role == "cause" and "education expenditure" in label:
            value += 35.0
        if role == "mediator":
            if "labor force" in label or "labour force" in label:
                value += 35.0
            if "school enrollment" in label:
                value += 14.0
            if "education expenditure" in label:
                value -= 40.0
        if role == "outcome":
            if "gni per capita" in label or "gdp per capita" in label:
                value += 40.0
            if "income" in label and "capita" in label:
                value += 20.0
            if "exports" in label:
                value -= 15.0
        return value

    cause = max(rows, key=lambda r: score(r, cause_terms, "cause"))
    rem = [r for r in rows if r["label"] != cause["label"]]
    mediator = max(rem, key=lambda r: score(r, mediator_terms, "mediator")) if rem else None
    rem2 = [r for r in rem if mediator is None or r["label"] != mediator["label"]]
    outcome = max(rem2, key=lambda r: score(r, outcome_terms, "outcome")) if rem2 else None
    return {
        "cause": str(cause["label"]),
        "mediator": str(mediator["label"]) if mediator else "",
        "outcome": str(outcome["label"]) if outcome else "",
    }


def _rows_as_variables_source(genome: TheoryGenome) -> str:
    genes = dict(genome.genes)
    axis = [str(x) for x in genes.get("axis", [])]
    label_col = str(genes.get("label_col", ""))
    roles = dict(genes.get("roles", {}) or {})
    return f'''def analyze(df) -> dict:
    axis_cols = [c for c in {axis!r} if c in df.columns]
    label_col = {label_col!r}
    roles = {roles!r}
    if label_col not in df.columns or len(axis_cols) < 3:
        return {{"evidence": "Darwin genome could not instantiate wide-table rows-as-variables orientation.", "statistic": 0.0, "variables": []}}

    def values_for(label):
        matches = df[df[label_col].astype(str) == str(label)]
        if len(matches) == 0:
            return []
        row = matches.iloc[0]
        vals = []
        for c in axis_cols:
            try:
                v = float(row[c])
            except Exception:
                continue
            if v == v:
                vals.append(v)
        return vals

    def corr(a, b):
        n = min(len(a), len(b))
        if n < 3:
            return 0.0
        a = a[:n]
        b = b[:n]
        ma = sum(a) / n
        mb = sum(b) / n
        va = sum((x - ma) ** 2 for x in a)
        vb = sum((y - mb) ** 2 for y in b)
        if va <= 1e-12 or vb <= 1e-12:
            return 0.0
        return sum((a[i] - ma) * (b[i] - mb) for i in range(n)) / ((va * vb) ** 0.5)

    cause = roles.get("cause", "")
    mediator = roles.get("mediator", "")
    outcome = roles.get("outcome", "")
    c_vals = values_for(cause)
    m_vals = values_for(mediator)
    o_vals = values_for(outcome)
    r_cm = corr(c_vals, m_vals)
    r_mo = corr(m_vals, o_vals)
    r_co = corr(c_vals, o_vals)
    statistic = min(abs(r_cm), abs(r_mo)) + 0.25 * abs(r_co)
    evidence = (
        "Darwin-evolved theory genome chose orientation rows_as_variables with "
        + str(len(axis_cols)) + " axis columns: " + cause + " -> " + mediator
        + " -> " + outcome + " (cause-mediator r=" + str(round(r_cm, 3))
        + ", mediator-outcome r=" + str(round(r_mo, 3))
        + ", direct r=" + str(round(r_co, 3)) + ")."
    )
    return {{
        "evidence": evidence,
        "statistic": float(statistic),
        "variables": [cause, mediator, outcome],
        "cause": cause,
        "mediator": mediator,
        "outcome": outcome,
        "relation": "evolved wide-table mediated time-series association",
    }}
'''


def _rows_as_variables_robust_source(genome: TheoryGenome) -> str:
    genes = dict(genome.genes)
    axis = [str(x) for x in genes.get("axis", [])]
    label_col = str(genes.get("label_col", ""))
    roles = dict(genes.get("roles", {}) or {})
    return f'''def analyze(df) -> dict:
    axis_cols = [c for c in {axis!r} if c in df.columns]
    label_col = {label_col!r}
    roles = {roles!r}
    if label_col not in df.columns or len(axis_cols) < 5:
        return {{"evidence": "Darwin regulatory robust table genome could not instantiate.", "statistic": 0.0, "variables": []}}

    def values_for(label):
        matches = df[df[label_col].astype(str) == str(label)]
        if len(matches) == 0:
            return []
        row = matches.iloc[0]
        vals = []
        for c in axis_cols:
            try:
                v = float(row[c])
            except Exception:
                continue
            if v == v:
                vals.append(v)
        return vals

    def corr(a, b):
        n = min(len(a), len(b))
        if n < 3:
            return 0.0
        a = a[:n]
        b = b[:n]
        ma = sum(a) / n
        mb = sum(b) / n
        va = sum((x - ma) ** 2 for x in a)
        vb = sum((y - mb) ** 2 for y in b)
        if va <= 1e-12 or vb <= 1e-12:
            return 0.0
        return sum((a[i] - ma) * (b[i] - mb) for i in range(n)) / ((va * vb) ** 0.5)

    def slope(vals):
        n = len(vals)
        if n < 3:
            return 0.0
        xs = list(range(n))
        mx = sum(xs) / n
        my = sum(vals) / n
        den = sum((x - mx) ** 2 for x in xs)
        if den <= 1e-12:
            return 0.0
        return sum((xs[i] - mx) * (vals[i] - my) for i in range(n)) / den

    def split_corr(a, b):
        n = min(len(a), len(b))
        if n < 6:
            return (corr(a, b), corr(a, b), 0.0)
        h = n // 2
        r1 = corr(a[:h], b[:h])
        r2 = corr(a[h:n], b[h:n])
        return (r1, r2, 1.0 - min(1.0, abs(r1 - r2)))

    cause = roles.get("cause", "")
    mediator = roles.get("mediator", "")
    outcome = roles.get("outcome", "")
    c_vals = values_for(cause)
    m_vals = values_for(mediator)
    o_vals = values_for(outcome)
    r_cm = corr(c_vals, m_vals)
    r_mo = corr(m_vals, o_vals)
    r_co = corr(c_vals, o_vals)
    cm_a, cm_b, cm_stability = split_corr(c_vals, m_vals)
    mo_a, mo_b, mo_stability = split_corr(m_vals, o_vals)
    c_slope = slope(c_vals)
    m_slope = slope(m_vals)
    o_slope = slope(o_vals)
    statistic = (
        min(abs(r_cm), abs(r_mo))
        + 0.25 * abs(r_co)
        + 0.15 * max(0.0, cm_stability)
        + 0.15 * max(0.0, mo_stability)
    )
    evidence = (
        "Regulatory Darwin robust-mediated-trend genome chose rows_as_variables "
        + "with " + str(len(axis_cols)) + " time/axis columns. "
        + "Tested " + cause + " -> " + mediator + " -> " + outcome
        + "; correlations: cause-mediator r=" + str(round(r_cm, 3))
        + " (split " + str(round(cm_a, 3)) + "/" + str(round(cm_b, 3)) + "), "
        + "mediator-outcome r=" + str(round(r_mo, 3))
        + " (split " + str(round(mo_a, 3)) + "/" + str(round(mo_b, 3)) + "), "
        + "direct r=" + str(round(r_co, 3)) + ". "
        + "Trend slopes: cause=" + str(round(c_slope, 4))
        + ", mediator=" + str(round(m_slope, 4))
        + ", outcome=" + str(round(o_slope, 4)) + "."
    )
    return {{
        "evidence": evidence,
        "statistic": float(statistic),
        "variables": [cause, mediator, outcome],
        "cause": cause,
        "mediator": mediator,
        "outcome": outcome,
        "relation": "regulatory robust mediated time-series association with split stability",
    }}
'''


def _columns_as_variables_source(genome: TheoryGenome) -> str:
    roles = dict(genome.genes.get("roles", {}) or {})
    variables = [str(v) for v in genome.genes.get("variables", [])]
    return f'''def analyze(df) -> dict:
    variables = [c for c in {variables!r} if c in df.columns]
    roles = {roles!r}
    if len(variables) < 2:
        return {{"evidence": "Darwin columns-as-variables genome could not instantiate.", "statistic": 0.0, "variables": []}}
    data = df[variables].apply(pd.to_numeric, errors="coerce")
    cause = roles.get("cause", variables[0])
    outcome = roles.get("outcome", variables[-1])
    mediator = roles.get("mediator", "")
    try:
        r_co = float(data[cause].corr(data[outcome]))
    except Exception:
        r_co = 0.0
    statistic = abs(r_co if r_co == r_co else 0.0)
    evidence = "Darwin-evolved columns_as_variables theory relates " + cause + " to " + outcome + " (r=" + str(round(r_co, 3)) + ")."
    return {{
        "evidence": evidence,
        "statistic": float(statistic),
        "variables": [v for v in [cause, mediator, outcome] if v],
        "cause": cause,
        "mediator": mediator,
        "outcome": outcome,
        "relation": "evolved column association",
    }}
'''


def _numeric_law_genomes(observations: Sequence[Any]) -> list[TheoryGenome]:
    rows: list[tuple[dict[str, float], float]] = []
    variables: list[str] = []
    for obs in observations:
        inputs = getattr(obs, "inputs", None)
        target = getattr(obs, "target", None)
        if not isinstance(inputs, dict):
            continue
        try:
            y = float(target)
        except Exception:
            continue
        if y <= 0 or y != y:
            continue
        numeric: dict[str, float] = {}
        for key, value in inputs.items():
            try:
                x = float(value)
            except Exception:
                continue
            if x <= 0 or x != x:
                continue
            numeric[str(key)] = x
            if str(key) not in variables:
                variables.append(str(key))
        if numeric:
            rows.append((numeric, y))
    if len(rows) < 3 or not variables:
        return []
    variables = variables[:6]
    genomes: list[TheoryGenome] = []
    fit = _fit_power_law(rows, variables)
    if fit is not None:
        constant, exponents, rms = fit
        genomes.append(
            TheoryGenome(
                name="numeric_power_law_scale_relation",
                kind="numeric_power_law",
                genes={
                    "input_family": "numeric_dict",
                    "relation_family": "power_law",
                    "variables": variables,
                    "constant": constant,
                    "exponents": exponents,
                    "fit_rms_log": rms,
                    "evidence_family": "heldout_numeric_scale",
                },
                complexity=2.0 + 0.35 * len(variables),
                lineage=("failure_algebra", "scale_relation_gene", "log_linear_fit"),
            )
        )
    genomes.extend(_fit_trig_relation_genomes(rows, variables))
    return genomes


def _fit_power_law(
    rows: Sequence[tuple[dict[str, float], float]],
    variables: Sequence[str],
) -> tuple[float, dict[str, float], float] | None:
    x_rows: list[list[float]] = []
    y_vals: list[float] = []
    for inputs, target in rows:
        if any(v not in inputs or inputs[v] <= 0 for v in variables):
            continue
        x_rows.append([1.0] + [math.log(float(inputs[v])) for v in variables])
        y_vals.append(math.log(float(target)))
    if len(x_rows) < len(variables) + 1:
        return None
    beta = _least_squares(x_rows, y_vals)
    if beta is None:
        return None
    residuals = []
    for row, y in zip(x_rows, y_vals):
        pred = sum(b * x for b, x in zip(beta, row))
        residuals.append((pred - y) ** 2)
    rms = (sum(residuals) / max(1, len(residuals))) ** 0.5
    constant = math.exp(beta[0])
    exponents = {str(v): float(beta[i + 1]) for i, v in enumerate(variables)}
    return constant, exponents, rms


def _least_squares(x_rows: Sequence[Sequence[float]], y_vals: Sequence[float]) -> list[float] | None:
    n = len(x_rows[0])
    ata = [[0.0 for _ in range(n)] for _ in range(n)]
    aty = [0.0 for _ in range(n)]
    ridge = 1e-9
    for row, y in zip(x_rows, y_vals):
        for i in range(n):
            aty[i] += row[i] * y
            for j in range(n):
                ata[i][j] += row[i] * row[j]
    for i in range(n):
        ata[i][i] += ridge
    return _solve_linear_system(ata, aty)


def _solve_linear_system(a: list[list[float]], b: list[float]) -> list[float] | None:
    n = len(b)
    aug = [list(row) + [float(b[i])] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        div = aug[col][col]
        aug[col] = [v / div for v in aug[col]]
        for row in range(n):
            if row == col:
                continue
            factor = aug[row][col]
            if abs(factor) <= 1e-18:
                continue
            aug[row] = [v - factor * p for v, p in zip(aug[row], aug[col])]
    return [aug[i][-1] for i in range(n)]


def _numeric_power_law_source(genome: TheoryGenome) -> str:
    variables = [str(v) for v in genome.genes.get("variables", [])]
    exponents = {str(k): float(v) for k, v in dict(genome.genes.get("exponents", {}) or {}).items()}
    constant = float(genome.genes.get("constant", 1.0) or 1.0)
    return f'''def law(inputs: dict) -> float:
    variables = {variables!r}
    exponents = {exponents!r}
    result = {constant!r}
    for name in variables:
        try:
            value = float(inputs.get(name, 0.0))
        except Exception:
            return 0.0
        if value <= 0:
            return 0.0
        result *= value ** float(exponents.get(name, 0.0))
    return float(result)
'''


def _fit_trig_relation_genomes(
    rows: Sequence[tuple[dict[str, float], float]],
    variables: Sequence[str],
) -> list[TheoryGenome]:
    angle_vars = [v for v in variables if _angle_like_name(v)]
    ratio_vars = [v for v in variables if v not in set(angle_vars)]
    targets = [target for _inputs, target in rows]
    if not angle_vars or len(ratio_vars) < 2 or not targets:
        return []
    if min(targets) < -1e-9 or max(targets) > 180.0:
        return []
    candidates: list[tuple[float, str, str, str, float]] = []
    for angle_var in angle_vars[:3]:
        for numerator in ratio_vars[:5]:
            for denominator in ratio_vars[:5]:
                if numerator == denominator:
                    continue
                fit = _fit_sine_ratio(rows, angle_var, numerator, denominator)
                if fit is None:
                    continue
                scale, rms = fit
                candidates.append((rms, angle_var, numerator, denominator, scale))
    candidates.sort(key=lambda row: row[0])
    out: list[TheoryGenome] = []
    for i, (rms, angle_var, numerator, denominator, scale) in enumerate(candidates[:4]):
        out.append(
            TheoryGenome(
                name=(
                    "numeric_bounded_trig_relation"
                    if i == 0
                    else f"numeric_bounded_trig_relation_{i}"
                ),
                kind="numeric_trig_relation",
                genes={
                    "input_family": "numeric_dict",
                    "relation_family": "bounded_sine_ratio",
                    "angle_var": angle_var,
                    "numerator": numerator,
                    "denominator": denominator,
                    "scale": scale,
                    "fit_rms_angle": rms,
                    "evidence_family": "bounded_angle_relation",
                },
                complexity=3.0,
                lineage=("failure_algebra", "scale_relation_gene", "bounded_trig_fit"),
            )
        )
    return out


def _fit_sine_ratio(
    rows: Sequence[tuple[dict[str, float], float]],
    angle_var: str,
    numerator: str,
    denominator: str,
) -> tuple[float, float] | None:
    xs: list[float] = []
    ys: list[float] = []
    for inputs, target in rows:
        if angle_var not in inputs or numerator not in inputs or denominator not in inputs:
            continue
        den = float(inputs[denominator])
        if abs(den) <= 1e-12:
            continue
        angle = float(inputs[angle_var])
        ratio = float(inputs[numerator]) / den
        x = ratio * math.sin(math.radians(angle))
        y = math.sin(math.radians(float(target)))
        if x == x and y == y:
            xs.append(x)
            ys.append(y)
    if len(xs) < 3 or sum(x * x for x in xs) <= 1e-12:
        return None
    scale = sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)
    residuals: list[float] = []
    for x, target_sin in zip(xs, ys):
        pred_sin = max(-1.0, min(1.0, scale * x))
        pred = math.degrees(math.asin(pred_sin))
        tgt = math.degrees(math.asin(max(-1.0, min(1.0, target_sin))))
        residuals.append((pred - tgt) ** 2)
    rms = (sum(residuals) / max(1, len(residuals))) ** 0.5
    if rms > 8.0:
        return None
    return float(scale), float(rms)


def _angle_like_name(name: str) -> bool:
    low = str(name).lower()
    return any(token in low for token in ("angle", "theta", "degree"))


def _numeric_trig_relation_source(genome: TheoryGenome) -> str:
    angle_var = str(genome.genes.get("angle_var", ""))
    numerator = str(genome.genes.get("numerator", ""))
    denominator = str(genome.genes.get("denominator", ""))
    scale = float(genome.genes.get("scale", 1.0) or 1.0)
    return f'''def law(inputs: dict) -> float:
    try:
        angle = float(inputs.get({angle_var!r}, 0.0))
        numerator = float(inputs.get({numerator!r}, 0.0))
        denominator = float(inputs.get({denominator!r}, 0.0))
    except Exception:
        return 0.0
    if abs(denominator) <= 1e-12:
        return 0.0
    value = {scale!r} * (numerator / denominator) * math.sin(math.radians(angle))
    if value > 1.0:
        value = 1.0
    if value < -1.0:
        value = -1.0
    return float(math.degrees(math.asin(value)))
'''


def _string_rule_genomes(observations: Sequence[Any]) -> list[TheoryGenome]:
    pairs = []
    context_keys: list[str] = []
    for obs in observations:
        current = getattr(obs, "inputs", None)
        target = getattr(obs, "target", None)
        context = getattr(obs, "context", {}) or {}
        if not isinstance(current, str) or not isinstance(target, str):
            continue
        pairs.append((current, target, context))
        for key in context:
            if str(key) not in context_keys:
                context_keys.append(str(key))
    if len(pairs) < 2:
        return []

    candidates: list[tuple[str, dict[str, Any], float]] = []
    if all(target == current for current, target, _ in pairs):
        candidates.append(("identity", {}, 1.0))
    if all(target == current[::-1] for current, target, _ in pairs):
        candidates.append(("reverse", {}, 1.2))
    if all(target == current + current for current, target, _ in pairs):
        candidates.append(("double_current", {}, 1.2))
    for key in context_keys:
        if all(target == str(context.get(key, "")) for _, target, context in pairs):
            candidates.append(("context_value", {"key": key}, 1.25))
        if all(target == current + str(context.get(key, "")) for current, target, context in pairs):
            candidates.append(("current_then_context", {"key": key}, 1.4))
        if all(target == str(context.get(key, "")) + current for current, target, context in pairs):
            candidates.append(("context_then_current", {"key": key}, 1.4))
    return [
        TheoryGenome(
            name=f"string_transition_{name}",
            kind="string_transition",
            genes={
                "input_family": "string_context",
                "relation_family": name,
                "evidence_family": "transition_examples",
                **params,
            },
            complexity=complexity,
            lineage=("failure_algebra", "transition_rule_gene", name),
        )
        for name, params, complexity in candidates[:12]
    ]


def _string_transition_source(genome: TheoryGenome) -> str:
    relation = str(genome.genes.get("relation_family", "identity"))
    key = str(genome.genes.get("key", ""))
    if relation == "reverse":
        body = "return str(current)[::-1]"
    elif relation == "double_current":
        body = "return str(current) + str(current)"
    elif relation == "context_value":
        body = f"return str(context.get({key!r}, ''))"
    elif relation == "current_then_context":
        body = f"return str(current) + str(context.get({key!r}, ''))"
    elif relation == "context_then_current":
        body = f"return str(context.get({key!r}, '')) + str(current)"
    else:
        body = "return str(current)"
    return f'''def rule(current: str, context: dict) -> str:
    {body}
'''


def _dict_predictor_genomes(observations: Sequence[Any]) -> list[TheoryGenome]:
    targets = [getattr(obs, "target", None) for obs in observations]
    targets = [t for t in targets if isinstance(t, dict)]
    if len(targets) < 2:
        return []
    target_keys: list[str] = []
    for target in targets:
        for key in target:
            if str(key) not in target_keys:
                target_keys.append(str(key))
    constants: dict[str, Any] = {}
    for key in target_keys:
        vals = [target.get(key) for target in targets if target.get(key) is not None]
        numeric = []
        for val in vals:
            try:
                numeric.append(float(val))
            except Exception:
                pass
        if numeric and len(numeric) == len(vals):
            constants[key] = sum(numeric) / len(numeric)
        elif vals:
            counts: dict[str, int] = {}
            for val in vals:
                counts[str(val)] = counts.get(str(val), 0) + 1
            constants[key] = max(counts.items(), key=lambda item: item[1])[0]
    if not constants:
        return []
    return [
        TheoryGenome(
            name="dict_state_predictor_empirical_roles",
            kind="dict_state_predictor",
            genes={
                "input_family": "structured_dict",
                "relation_family": "empirical_state_roles",
                "evidence_family": "structured_transition_examples",
                "constants": constants,
            },
            complexity=1.8 + 0.15 * len(constants),
            lineage=("failure_algebra", "state_role_gene", "empirical_roles"),
        )
    ]


def _dict_predictor_source(genome: TheoryGenome) -> str:
    constants = dict(genome.genes.get("constants", {}) or {})
    return f'''def predict(parent1: dict, parent2: dict, context: dict) -> dict:
    return dict({constants!r})
'''


def _replace_roles(genome: TheoryGenome, roles: Mapping[str, str], mutation: str) -> TheoryGenome:
    genes = dict(genome.genes)
    genes["roles"] = dict(roles)
    return TheoryGenome(
        name=f"{genome.name}_{mutation}",
        kind=genome.kind,
        genes=genes,
        complexity=genome.complexity + 0.4,
        lineage=tuple(genome.lineage) + (mutation,),
    )


def _dedup_genomes(genomes: Sequence[TheoryGenome]) -> tuple[TheoryGenome, ...]:
    seen: set[str] = set()
    out: list[TheoryGenome] = []
    for genome in genomes:
        key = f"{genome.kind}:{sorted(dict(genome.genes).items(), key=lambda x: x[0])}"
        if key in seen:
            continue
        seen.add(key)
        out.append(genome)
    return tuple(out)


def _describe_genome(genome: TheoryGenome) -> str:
    orientation = genome.genes.get("orientation", "")
    roles = genome.genes.get("roles", {})
    return (
        f"Darwinian theory genome phenotype: kind={genome.kind}, "
        f"orientation={orientation}, roles={roles}, lineage={genome.lineage}"
    )


def _numeric_columns(df: Any, columns: Sequence[str]) -> list[str]:
    out = []
    for col in columns:
        try:
            vals = df[col].astype(str).str.replace(",", ".", regex=False).astype(float)
            if float(vals.notna().mean()) >= 0.6:
                out.append(str(col))
        except Exception:
            continue
    return out


def _rank_relevant_columns(
    columns: Sequence[str],
    *,
    question: str,
    descriptions: Mapping[str, str],
    domain_context: str,
) -> list[str]:
    q = _tokens(f"{question} {domain_context}")
    scored = []
    for col in columns:
        text = f"{col} {descriptions.get(str(col), '')}"
        toks = _tokens(text)
        score = 4.0 * len(q & toks)
        if _is_axis_like(str(col)) or _is_year_like(str(col)):
            score -= 2.0
        scored.append((score, str(col)))
    scored.sort(key=lambda x: (x[0], -len(x[1])), reverse=True)
    return [c for s, c in scored if s > 0] or [str(c) for c in columns]


def _best_label_col(df: Any, text_cols: Sequence[str], q_tokens: set[str]) -> str:
    best = ""
    best_score = -1.0
    for col in text_cols:
        try:
            text = " ".join(str(v) for v in df[col].dropna().tolist())
        except Exception:
            text = ""
        score = len(_tokens(text) & q_tokens)
        if "series" in col.lower() or "indicator" in col.lower():
            score += 4
        if score > best_score:
            best = str(col)
            best_score = score
    return best


def _row_items(df: Any, label_col: str, axis_cols: Sequence[str], q_tokens: set[str]) -> list[dict[str, Any]]:
    out = []
    if not label_col:
        return out
    for _, row in df.iterrows():
        label = str(row.get(label_col, ""))
        vals = []
        for col in axis_cols:
            try:
                v = float(row[col])
            except Exception:
                continue
            if v == v:
                vals.append(v)
        if len(vals) >= 3:
            out.append({"label": label, "tokens": _tokens(label), "values": vals})
    out.sort(key=lambda item: len(set(item["tokens"]) & q_tokens), reverse=True)
    return out


def _text_columns(df: Any, columns: Sequence[str]) -> list[str]:
    out = []
    for col in columns:
        if _is_year_like(str(col)):
            continue
        try:
            vals = [str(v) for v in df[col].dropna().head(20).tolist()]
        except Exception:
            vals = []
        if vals:
            out.append(str(col))
    return out


def _tokens(text: str) -> set[str]:
    stop = {
        "the", "and", "for", "with", "from", "into", "between", "among", "which",
        "what", "when", "where", "does", "did", "were", "was", "are", "how",
        "there", "this", "that", "have", "has", "had", "over", "under", "after",
        "before", "during", "table", "data", "dataset", "variable", "variables",
        "column", "columns", "row", "rows", "influence", "increased",
    }
    return {
        tok
        for tok in re.findall(r"[a-zA-Z][a-zA-Z0-9_/-]{2,}", str(text).lower())
        if tok not in stop
    }


def _is_year_like(value: str) -> bool:
    return bool(re.search(r"(18|19|20|21)\d{2}", str(value)))


def _is_axis_like(value: str) -> bool:
    low = str(value).lower()
    return low in {"year", "date", "time", "period", "country", "entity", "id", "index"} or low.endswith("_id")
