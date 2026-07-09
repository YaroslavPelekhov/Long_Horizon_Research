"""Role canonicalization for executable estimands.

The canonicalizer maps observable columns into semantic roles before a
statistical functional is fit.  This keeps the estimand layer from treating
neighboring proxy columns as different hypotheses when they are really noisy
measurements of the same role, such as multiple urban-land-use years.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Mapping


@dataclass(frozen=True)
class CanonicalRole:
    name: str
    label: str
    kind: str
    columns: tuple[str, ...]
    score: float
    evidence: str


@dataclass(frozen=True)
class MaterializedRole:
    name: str
    label: str
    column: str
    source_columns: tuple[str, ...]
    kind: str
    score: float
    evidence: str


@dataclass(frozen=True)
class RoleSpec:
    name: str
    label: str
    kind: str
    aliases: tuple[str, ...]
    excludes: tuple[str, ...] = ()


ROLE_SPECS: tuple[RoleSpec, ...] = (
    RoleSpec("urban_land_use", "urban land use", "numeric", ("urban", "urban land", "city", "built")),
    RoleSpec("cropland_land_use", "cropland land use", "numeric", ("cropland", "crop land", "agriculture land")),
    RoleSpec("elevation", "elevation", "numeric", ("elevation", "altitude")),
    RoleSpec("introduction_pathway", "introduction pathways", "categorical", ("intro pathway", "introduction pathway", "pathway")),
    RoleSpec("residence_time", "minimum residence time", "numeric", ("mrt", "minimum residence", "residence time")),
    RoleSpec("socioeconomic_status", "socioeconomic status", "numeric", ("ses", "socioeconomic", "income", "parent education")),
    RoleSpec("race", "race", "categorical", ("race", "racial", "ethnic")),
    RoleSpec("gender", "gender", "categorical", ("gender", "sex")),
    RoleSpec("academic_ability", "academic characteristics", "numeric", ("academic", "ability", "asvab", "class percentile")),
    RoleSpec("wealth", "wealth", "numeric", ("wealth", "net wealth")),
    RoleSpec("criminal_history", "criminal history", "binary", ("criminal", "incarcerat", "jail", "convicted", "sentenced", "detention")),
    RoleSpec("degree_completion", "degree completion", "binary", ("degree completion", "ba degree", "completed")),
)


def select_interaction_roles(
    *,
    task_text: str,
    domain_context: str,
    df: Any,
    schema: Mapping[str, Any],
    target: str,
) -> tuple[MaterializedRole, ...]:
    """Return materialized role features for interaction fitting."""

    roles = canonicalize_roles(
        task_text=f"{task_text}\n{domain_context}",
        df=df,
        schema=schema,
        exclude_columns=(target,),
    )
    selected: list[MaterializedRole] = []
    for role in roles:
        materialized = materialize_role(df, role)
        if materialized is None:
            continue
        if materialized.column == target:
            continue
        selected.append(materialized)
        if len(selected) >= 3:
            break
    return tuple(selected)


def canonicalize_roles(
    *,
    task_text: str,
    df: Any,
    schema: Mapping[str, Any],
    exclude_columns: tuple[str, ...] = (),
) -> tuple[CanonicalRole, ...]:
    """Cluster schema columns into query-supported semantic roles."""

    if df is None or not hasattr(df, "columns"):
        return ()
    support_text = str(task_text or "")
    support_tokens = _tokens(support_text)
    excluded = {str(c) for c in exclude_columns}
    roles: list[CanonicalRole] = []
    for spec in ROLE_SPECS:
        query_score = _alias_score(spec.aliases, support_text, support_tokens)
        if query_score <= 0:
            continue
        columns: list[tuple[float, str]] = []
        for col in getattr(df, "columns", []):
            col_s = str(col)
            if col_s in excluded or col_s.startswith("__estimand_target"):
                continue
            text = _column_text(col_s, schema)
            low = text.lower()
            if any(ex in low for ex in spec.excludes):
                continue
            alias = _alias_score(spec.aliases, text, _tokens(text))
            if alias <= 0:
                continue
            shape = _shape_score(df, col_s, spec.kind)
            if shape <= -2:
                continue
            score = 4.0 * query_score + 3.0 * alias + shape + _recency_score(col_s)
            columns.append((score, col_s))
        if not columns:
            continue
        columns.sort(key=lambda item: item[0], reverse=True)
        kept = _dedupe_role_columns(columns, spec)
        evidence = ",".join(f"{col}:{score:.2f}" for score, col in kept[:6])
        roles.append(
            CanonicalRole(
                name=spec.name,
                label=spec.label,
                kind=spec.kind,
                columns=tuple(col for _score, col in kept),
                score=max(score for score, _col in kept),
                evidence=evidence,
            )
        )
    roles.extend(_induce_dynamic_roles(support_text, support_tokens, df, schema, excluded, roles))
    roles.sort(key=lambda role: role.score, reverse=True)
    return tuple(roles)


def materialize_role(df: Any, role: CanonicalRole) -> MaterializedRole | None:
    """Create one table column representing a role."""

    if not role.columns:
        return None
    if len(role.columns) == 1 or role.kind in {"categorical", "binary"}:
        col = role.columns[0]
        return MaterializedRole(
            name=role.name,
            label=role.label,
            column=col,
            source_columns=role.columns,
            kind=role.kind,
            score=role.score,
            evidence=role.evidence,
        )
    import pandas as pd

    parts = []
    for col in role.columns:
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().sum() < max(8, len(df) // 20):
            continue
        std = float(values.std(skipna=True) or 0.0)
        if not math.isfinite(std) or std <= 1e-12:
            continue
        parts.append((values - float(values.mean(skipna=True))) / std)
    if not parts:
        return None
    out = f"__role_{role.name}"
    df[out] = pd.concat(parts, axis=1).mean(axis=1, skipna=True)
    return MaterializedRole(
        name=role.name,
        label=role.label,
        column=out,
        source_columns=role.columns,
        kind=role.kind,
        score=role.score,
        evidence=role.evidence,
    )


def _dedupe_role_columns(columns: list[tuple[float, str]], spec: RoleSpec) -> list[tuple[float, str]]:
    if spec.kind in {"categorical", "binary"}:
        return columns[:1]
    out: list[tuple[float, str]] = []
    seen_base: set[str] = set()
    for score, col in columns:
        base = _base_measure_name(col)
        if base in seen_base and not _has_year(col):
            continue
        seen_base.add(base)
        out.append((score, col))
        if len(out) >= 8:
            break
    return out


def _induce_dynamic_roles(
    support_text: str,
    support_tokens: set[str],
    df: Any,
    schema: Mapping[str, Any],
    excluded: set[str],
    existing: list[CanonicalRole],
) -> list[CanonicalRole]:
    """Induce schema/query-specific roles without benchmark-specific names."""

    clusters: dict[str, list[tuple[float, str]]] = {}
    labels: dict[str, str] = {}
    kinds: dict[str, list[str]] = {}
    existing_column_sets = [set(role.columns) for role in existing]
    for col in getattr(df, "columns", []):
        col_s = str(col)
        if col_s in excluded or col_s.startswith("__estimand_target"):
            continue
        text = _column_text(col_s, schema)
        col_tokens = _tokens(text)
        overlap = support_tokens & col_tokens
        if not overlap:
            continue
        filtered = tuple(sorted(tok for tok in overlap if not _generic_token(tok)))
        if not filtered:
            continue
        key_tokens = filtered[:3]
        key = "_".join(key_tokens)
        label = _dynamic_label(key_tokens, support_text)
        shape_numeric = _shape_score(df, col_s, "numeric")
        shape_categorical = _shape_score(df, col_s, "categorical")
        kind = "numeric" if shape_numeric >= shape_categorical else "categorical"
        shape = max(shape_numeric, shape_categorical)
        if shape <= -2:
            continue
        score = 2.0 * len(filtered) + 1.5 * _phrase_overlap(label, text) + shape + _recency_score(col_s)
        clusters.setdefault(key, []).append((score, col_s))
        labels.setdefault(key, label)
        kinds.setdefault(key, []).append(kind)

    roles: list[CanonicalRole] = []
    for key, columns in clusters.items():
        columns.sort(key=lambda item: item[0], reverse=True)
        kept = columns[:8]
        col_set = {col for _score, col in kept}
        if any(col_set and col_set <= old for old in existing_column_sets):
            continue
        if len(kept) == 1 and kept[0][0] < 5.0:
            continue
        kind = "numeric" if kinds.get(key, []).count("numeric") >= kinds.get(key, []).count("categorical") else "categorical"
        label = labels.get(key, key.replace("_", " "))
        evidence = ",".join(f"{col}:{score:.2f}" for score, col in kept[:6])
        roles.append(
            CanonicalRole(
                name=f"dynamic_{key}",
                label=label,
                kind=kind,
                columns=tuple(col for _score, col in kept),
                score=max(score for score, _col in kept) - 0.5,
                evidence=evidence,
            )
        )
    return roles


def _alias_score(aliases: tuple[str, ...], text: str, tokens: set[str]) -> float:
    low = str(text or "").lower().replace("_", " ").replace(".", " ")
    score = 0.0
    for alias in aliases:
        alias_l = alias.lower()
        alias_tokens = _tokens(alias_l)
        if alias_l in low:
            score += 2.5
        elif alias_tokens and alias_tokens <= tokens:
            score += 2.0
        elif alias_tokens and alias_tokens & tokens:
            score += len(alias_tokens & tokens) / max(1, len(alias_tokens))
    return score


def _phrase_overlap(phrase: str, text: str) -> float:
    phrase_tokens = _tokens(phrase)
    if not phrase_tokens:
        return 0.0
    return len(phrase_tokens & _tokens(text)) / max(1, len(phrase_tokens))


def _dynamic_label(tokens: tuple[str, ...], support_text: str) -> str:
    low = str(support_text or "").lower()
    for n in range(min(4, len(tokens) + 2), 0, -1):
        words = re.findall(r"[a-zA-Z][a-zA-Z0-9-]{1,}", low)
        for i in range(0, max(0, len(words) - n + 1)):
            phrase = " ".join(words[i : i + n])
            phrase_tokens = _tokens(phrase)
            if set(tokens) <= phrase_tokens:
                return phrase
    return " ".join(tokens)


def _generic_token(token: str) -> bool:
    return token in {
        "factor", "factors", "interact", "interaction", "significant", "affect",
        "effect", "proportion", "rate", "rates", "relationship", "compare",
        "compared", "level", "levels", "value", "values", "column", "data",
        "table", "outcome", "predictor", "group", "between", "question",
    }


def _shape_score(df: Any, col: str, kind: str) -> float:
    try:
        import pandas as pd

        series = df[col]
        numeric = pd.to_numeric(series, errors="coerce")
        n_numeric = int(numeric.notna().sum())
        nunique = int(series.nunique(dropna=True))
    except Exception:
        return 0.0
    if kind == "numeric":
        if n_numeric < max(8, len(df) // 20):
            return -3.0
        std = float(numeric.std(skipna=True) or 0.0)
        return 1.0 if std > 0 else -3.0
    if kind == "categorical":
        return 1.0 if 1 < nunique <= max(40, len(df) // 3) else -2.0
    if kind == "binary":
        values = set(numeric.dropna().unique().tolist())
        return 1.0 if values and values <= {0, 1, 0.0, 1.0} else 0.0
    return 0.0


def _recency_score(col: str) -> float:
    years = [int(x) for x in re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", str(col))]
    if not years:
        return 0.0
    return min(2.0, max(years) / 3000.0)


def _column_text(col: str, schema: Mapping[str, Any]) -> str:
    return f"{col} {schema.get(str(col), '')}"


def _base_measure_name(col: str) -> str:
    low = str(col).lower()
    low = re.sub(r"(?<!\d)((?:19|20)\d{2})(?!\d)", "", low)
    low = re.sub(r"\b\d+(?:m|km)?\b", "", low)
    low = re.sub(r"[^a-z]+", " ", low)
    return " ".join(low.split())


def _has_year(col: str) -> bool:
    return bool(re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", str(col)))


def _tokens(text: str) -> set[str]:
    stop = {
        "the", "and", "for", "with", "from", "into", "between", "among", "which",
        "what", "when", "where", "does", "did", "were", "was", "are", "how",
        "there", "this", "that", "have", "has", "had", "over", "under", "after",
        "before", "during", "in", "of", "to", "a", "an", "on", "by", "as",
        "using", "use", "used", "their", "they", "who", "while", "respectively",
    }
    out: set[str] = set()
    for raw in re.findall(r"[a-zA-Z][a-zA-Z0-9_.-]{1,}", str(text or "").lower()):
        for token in re.split(r"[^a-zA-Z0-9]+", raw):
            if len(token) <= 2 or token in stop:
                continue
            out.add(token)
            if token.endswith("s") and len(token) > 4:
                out.add(token[:-1])
    return out
