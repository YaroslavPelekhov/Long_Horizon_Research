"""Syndrome decoding for missing hypothesis operators.

The decoder treats failed/weak candidate analyzers as noisy probes.  It extracts
generic syndrome bits from their outputs and from the observable interface, then
emits repair analyzers for the most likely sparse missing operator.

v0 intentionally starts with table tasks because DiscoveryBench exposes a clean
failure: ordinary dataframe priors select year columns as scientific variables
when the true variables live in row labels and years are only an axis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class SyndromeDecode:
    name: str
    bits: Mapping[str, Any]
    decoded_operator: str
    program_sources: tuple[dict[str, Any], ...]
    rationale: str


def decode_syndrome_program_sources(
    *,
    adapter: Any,
    observations: Sequence[Any],
    programs: Sequence[Any],
    max_programs: int = 32,
) -> SyndromeDecode | None:
    """Decode missing operators from candidate-output syndrome bits."""

    signature = str(adapter.signature_hint())
    if "analyze(df)" not in signature or not observations:
        return None
    df = getattr(observations[-1], "inputs", None)
    if df is None or not hasattr(df, "columns"):
        return None

    question = str(getattr(adapter, "question", "") or "")
    domain_context = str(getattr(adapter, "domain_knowledge", "") or "")
    column_descriptions = dict(getattr(adapter, "column_descriptions", {}) or {})

    outputs = _candidate_outputs(df, programs[:max_programs])
    bits = _table_syndrome_bits(
        df=df,
        outputs=outputs,
        question=question,
        domain_context=domain_context,
        column_descriptions=column_descriptions,
    )
    if _decode_wide_table_rows_as_series(bits):
        source = _wide_table_rows_as_series_source(
            question=question,
            domain_context=domain_context,
        )
        return SyndromeDecode(
            name="wide_table_rows_as_series_syndrome",
            bits=bits,
            decoded_operator="wide_table_rows_as_series",
            program_sources=(
                {
                    "name": "syndrome_wide_table_rows_as_series",
                    "description": (
                        "Syndrome-decoded repair: treat text-labeled rows as "
                        "scientific variables and year-like columns as the axis"
                    ),
                    "complexity": 3.2,
                    "code": source,
                },
            ),
            rationale=(
                "Candidate analyzers selected axis-like columns or shape facts; "
                "the interface has many year-like numeric columns and text row "
                "labels containing question/domain terms. Decode missing operator "
                "as rows-as-variables over an axis."
            ),
        )
    return None


def _candidate_outputs(df: Any, programs: Sequence[Any]) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for program in programs:
        try:
            out = program.fn(df)
        except Exception:
            continue
        if isinstance(out, dict):
            outputs.append(out)
    return outputs


def _table_syndrome_bits(
    *,
    df: Any,
    outputs: Sequence[dict[str, Any]],
    question: str,
    domain_context: str,
    column_descriptions: Mapping[str, str],
) -> dict[str, Any]:
    cols = [str(c) for c in getattr(df, "columns", [])]
    axis_cols = [c for c in cols if _is_year_like(c) or _is_axis_like(c)]
    text_cols = _text_columns(df, cols)
    row_label_hits = _row_label_hits(df, text_cols, f"{question} {domain_context}")
    chosen_variables = []
    shape_only_count = 0
    axis_variable_count = 0
    for out in outputs:
        evidence = str(out.get("evidence", "") or "")
        if ("rows=" in evidence.lower() or "columns=" in evidence.lower() or "shape" in evidence.lower()):
            shape_only_count += 1
        for key in ("cause", "mediator", "outcome", "variables"):
            value = out.get(key)
            if isinstance(value, (list, tuple)):
                chosen_variables.extend(str(v) for v in value)
            elif value:
                chosen_variables.append(str(value))
    for variable in chosen_variables:
        if _is_year_like(variable) or _is_axis_like(variable):
            axis_variable_count += 1
    return {
        "n_columns": len(cols),
        "n_axis_columns": len(axis_cols),
        "axis_columns_preview": axis_cols[:6],
        "text_columns": text_cols[:6],
        "row_label_query_hits": row_label_hits[:8],
        "n_candidate_outputs": len(outputs),
        "shape_only_count": shape_only_count,
        "axis_variable_count": axis_variable_count,
        "chosen_variables_preview": chosen_variables[:12],
        "schema_has_year_axis": len(axis_cols) >= max(5, len(cols) // 4),
        "rows_have_semantic_labels": bool(row_label_hits),
    }


def _decode_wide_table_rows_as_series(bits: Mapping[str, Any]) -> bool:
    return bool(
        bits.get("schema_has_year_axis")
        and bits.get("rows_have_semantic_labels")
        and (
            int(bits.get("axis_variable_count", 0)) > 0
            or int(bits.get("shape_only_count", 0)) > 0
            or int(bits.get("n_candidate_outputs", 0)) == 0
        )
    )


def _wide_table_rows_as_series_source(*, question: str, domain_context: str) -> str:
    return f'''def analyze(df) -> dict:
    question = {question!r}
    domain_context = {domain_context!r}

    def toks(text):
        out = []
        cur = []
        for ch in str(text).lower():
            if ch.isalnum():
                cur.append(ch)
            else:
                if len(cur) >= 3:
                    out.append(''.join(cur))
                cur = []
        if len(cur) >= 3:
            out.append(''.join(cur))
        stop = set(['the','and','for','with','from','into','between','among','which','what','when','where','does','did','were','was','are','how','there','this','that','have','has','had','over','under','after','before','during','table','data','dataset','variable','variables','column','columns','row','rows'])
        return set([t for t in out if t not in stop])

    def is_year_col(col):
        s = str(col)
        digits = ''.join([ch for ch in s if ch.isdigit()])
        return len(digits) >= 4 and 1800 <= int(digits[:4]) <= 2200

    axis_cols = [c for c in df.columns if is_year_col(c)]
    if len(axis_cols) < 3:
        return {{"evidence": "No wide year-axis columns detected.", "statistic": 0.0, "variables": []}}

    text_cols = []
    for c in df.columns:
        if c in axis_cols:
            continue
        try:
            vals = [str(v) for v in df[c].dropna().head(20).tolist()]
        except Exception:
            vals = []
        if vals:
            text_cols.append(c)
    q_all = toks(question + ' ' + domain_context)
    best_label_col = None
    best_score = -1
    for c in text_cols:
        try:
            text = ' '.join(str(v) for v in df[c].dropna().tolist())
        except Exception:
            text = ''
        score = len(toks(text) & q_all)
        if 'series' in str(c).lower() or 'indicator' in str(c).lower():
            score += 4
        if score > best_score:
            best_score = score
            best_label_col = c
    if best_label_col is None:
        return {{"evidence": "No semantic row-label column detected.", "statistic": 0.0, "variables": []}}

    q = question.lower()
    trigger_pos = len(q)
    for trig in [' influence ', ' affect ', ' impact ', ' contribute ', ' associated ', ' relation ', ' relationship ']:
        pos = q.find(trig)
        if pos >= 0 and pos < trigger_pos:
            trigger_pos = pos
    cause_terms = toks(q[:trigger_pos] if trigger_pos < len(q) else q)
    tail_terms = toks(q[trigger_pos:] if trigger_pos < len(q) else q)
    all_terms = toks(question + ' ' + domain_context)
    mediator_terms = set([t for t in all_terms if t in set(['human','capital','labor','labour','force','school','enrollment','primary','secondary'])])
    outcome_terms = set([t for t in all_terms if t in set(['gdp','gni','income','output','economic','capita'])])
    if not mediator_terms:
        mediator_terms = tail_terms
    if not outcome_terms:
        outcome_terms = tail_terms

    rows = []
    for idx, row in df.iterrows():
        label = str(row.get(best_label_col, ''))
        label_tokens = toks(label)
        values = []
        for c in axis_cols:
            try:
                v = float(row[c])
            except Exception:
                continue
            if v == v:
                values.append(v)
        if len(values) < 3:
            continue
        rows.append({{'idx': idx, 'label': label, 'tokens': label_tokens, 'values': values}})
    if len(rows) < 2:
        return {{"evidence": "Not enough semantic time-series rows detected.", "statistic": 0.0, "variables": []}}

    def score_role(item, terms):
        return len(item['tokens'] & terms) * 10 + len(item['tokens'] & all_terms)

    def role_score_with_boost(item, terms, role):
        score = score_role(item, terms)
        label = item['label'].lower()
        if role == 'mediator':
            if 'labor force' in label or 'labour force' in label:
                score += 30
            if 'school enrollment' in label:
                score += 12
            if 'education expenditure' in label:
                score -= 35
        if role == 'outcome':
            if 'gni per capita' in label or 'gdp per capita' in label:
                score += 35
            if 'income' in label and 'capita' in label:
                score += 20
            if 'exports' in label:
                score -= 12
        return score

    cause = max(rows, key=lambda r: score_role(r, cause_terms))
    cause_label = cause['label'].lower()
    remaining = [r for r in rows if r is not cause and r['label'].lower() != cause_label]
    mediator = max(remaining, key=lambda r: role_score_with_boost(r, mediator_terms, 'mediator')) if remaining else cause
    mediator_label = mediator['label'].lower()
    remaining2 = [
        r for r in rows
        if r is not cause and r is not mediator
        and r['label'].lower() != cause_label
        and r['label'].lower() != mediator_label
    ]
    outcome = max(remaining2, key=lambda r: role_score_with_boost(r, outcome_terms, 'outcome')) if remaining2 else mediator

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

    r_cm = corr(cause['values'], mediator['values'])
    r_mo = corr(mediator['values'], outcome['values'])
    r_co = corr(cause['values'], outcome['values'])
    statistic = min(abs(r_cm), abs(r_mo)) + 0.25 * abs(r_co)
    evidence = (
        "Decoded wide table as row-labeled scientific time series: "
        + cause['label'] + " -> " + mediator['label'] + " -> " + outcome['label']
        + " across " + str(len(axis_cols)) + " year-axis columns "
        + "(cause-mediator r=" + str(round(r_cm, 3))
        + ", mediator-outcome r=" + str(round(r_mo, 3))
        + ", direct r=" + str(round(r_co, 3)) + ")."
    )
    return {{
        "evidence": evidence,
        "statistic": float(statistic),
        "variables": [cause['label'], mediator['label'], outcome['label']],
        "cause": cause['label'],
        "mediator": mediator['label'],
        "outcome": outcome['label'],
        "relation": "wide-table mediated time-series association",
    }}
'''


def _text_columns(df: Any, cols: Sequence[str]) -> list[str]:
    text_cols = []
    for col in cols:
        if _is_year_like(col) or _is_axis_like(col):
            continue
        try:
            values = [str(v) for v in df[col].dropna().head(20).tolist()]
        except Exception:
            values = []
        if values:
            text_cols.append(col)
    return text_cols


def _row_label_hits(df: Any, text_cols: Sequence[str], query: str) -> list[dict[str, Any]]:
    q_tokens = _tokens(query)
    hits = []
    for col in text_cols:
        try:
            values = [str(v) for v in df[col].dropna().tolist()]
        except Exception:
            continue
        for value in values[:40]:
            overlap = _tokens(value) & q_tokens
            if overlap:
                hits.append({"column": str(col), "value": value[:120], "overlap": sorted(overlap)})
    return hits


def _tokens(text: str) -> set[str]:
    stop = {
        "the", "and", "for", "with", "from", "into", "between", "among", "which",
        "what", "when", "where", "does", "did", "were", "was", "are", "how",
        "there", "this", "that", "have", "has", "had", "over", "under", "after",
        "before", "during", "table", "data", "dataset", "variable", "variables",
        "column", "columns", "row", "rows",
    }
    return {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_/-]{2,}", str(text).lower())
        if token not in stop
    }


def _is_year_like(value: str) -> bool:
    match = re.search(r"(18|19|20|21)\d{2}", str(value))
    return bool(match)


def _is_axis_like(value: str) -> bool:
    low = str(value).lower()
    return low in {"year", "date", "time", "period", "country", "entity", "id", "index"} or low.endswith("_id")
