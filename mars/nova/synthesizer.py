"""NOVA analyzer synthesis from local oracle supervision."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

from mars.nova.oracle_tasks import OracleCurriculum, build_oracle_curriculum
from mars.nova.prior import AnalyzerSketch, HypothesisPrior


@dataclass(frozen=True)
class NOVAResult:
    """Candidate sources and local-prior trace emitted by NOVA."""

    program_sources: tuple[dict[str, Any], ...]
    curriculum: OracleCurriculum
    trace: tuple[dict[str, Any], ...]


class NOVASynthesizer:
    """Build and train a local hypothesis prior, then emit analyzers."""

    def __init__(
        self,
        *,
        max_programs: int = 24,
        temperature: float = 0.25,
        complexity_weight: float = 0.03,
    ):
        self.max_programs = int(max_programs)
        self.prior = HypothesisPrior(
            temperature=temperature,
            complexity_weight=complexity_weight,
        )

    def synthesize(
        self,
        *,
        signature_hint: str,
        observations: Sequence[Any],
        interface_name: str = "unknown",
        question: str = "",
        column_descriptions: Mapping[str, str] | None = None,
        domain_context: str = "",
    ) -> NOVAResult:
        curriculum = build_oracle_curriculum(
            observations,
            interface_name=interface_name,
        )
        sketches = self._sketches(
            signature_hint,
            curriculum,
            question=question,
            column_descriptions=column_descriptions or {},
            domain_context=domain_context,
        )
        fit = self.prior.fit(sketches, curriculum.tasks)
        sources = []
        for sketch, loss, posterior in fit.ranked[: self.max_programs]:
            item = {
                "name": sketch.name,
                "description": (
                    f"{sketch.description}; NOVA local-oracle loss={loss:.4f}, "
                    f"posterior={posterior:.4f}"
                ),
                "complexity": sketch.complexity,
                "code": sketch.code,
                "nova_loss": loss,
                "nova_posterior": posterior,
                "tags": list(sketch.tags),
            }
            sources.append(item)
        return NOVAResult(
            program_sources=tuple(sources),
            curriculum=curriculum,
            trace=fit.trace,
        )

    def _sketches(
        self,
        signature_hint: str,
        curriculum: OracleCurriculum,
        *,
        question: str = "",
        column_descriptions: Mapping[str, str] | None = None,
        domain_context: str = "",
    ) -> list[AnalyzerSketch]:
        sig = signature_hint.replace(" ", "")
        if "current:str" in sig and "context:dict" in sig:
            return _string_rule_sketches(curriculum)
        if "inputs:dict" in sig and "float" in sig:
            return _numeric_law_sketches(curriculum)
        if "analyze(df)" in signature_hint:
            return _table_analyzer_sketches(
                curriculum,
                question=question,
                column_descriptions=column_descriptions or {},
                domain_context=domain_context,
            )
        return []


def _string_rule_sketches(curriculum: OracleCurriculum) -> list[AnalyzerSketch]:
    tasks = curriculum.tasks
    sketches: list[AnalyzerSketch] = [
        AnalyzerSketch(
            name="nova_string_identity",
            description="NOVA prior candidate: return input string unchanged",
            code="def rule(current: str, context: dict) -> str:\n    return current\n",
            predict=lambda inputs, context: str(inputs),
            complexity=1.0,
        ),
        AnalyzerSketch(
            name="nova_string_reverse",
            description="NOVA prior candidate: reverse input string",
            code="def rule(current: str, context: dict) -> str:\n    return current[::-1]\n",
            predict=lambda inputs, context: str(inputs)[::-1],
            complexity=1.2,
        ),
    ]
    if tasks and all(isinstance(t.inputs, str) and isinstance(t.target, str) for t in tasks):
        suffix = _common_suffix_after_input(tasks)
        if suffix is not None:
            suffix_lit = repr(suffix)
            safe = _safe_name("nova_append_" + (suffix or "empty"))
            sketches.append(
                AnalyzerSketch(
                    name=safe,
                    description="NOVA learned a shared target-minus-input suffix from local oracle tasks",
                    code=(
                        "def rule(current: str, context: dict) -> str:\n"
                        f"    return current + {suffix_lit}\n"
                    ),
                    predict=lambda inputs, context, s=suffix: str(inputs) + s,
                    complexity=1.1 + min(1.0, len(suffix) / 12.0),
                )
            )
        prefix = _common_prefix_before_input(tasks)
        if prefix is not None:
            prefix_lit = repr(prefix)
            sketches.append(
                AnalyzerSketch(
                    name=_safe_name("nova_prepend_" + (prefix or "empty")),
                    description="NOVA learned a shared target-minus-input prefix from local oracle tasks",
                    code=(
                        "def rule(current: str, context: dict) -> str:\n"
                        f"    return {prefix_lit} + current\n"
                    ),
                    predict=lambda inputs, context, p=prefix: p + str(inputs),
                    complexity=1.1 + min(1.0, len(prefix) / 12.0),
                )
            )
        constant = _constant_target(tasks)
        if constant is not None:
            sketches.append(
                AnalyzerSketch(
                    name=_safe_name("nova_constant_" + constant[:24]),
                    description="NOVA learned a constant output from local oracle tasks",
                    code=(
                        "def rule(current: str, context: dict) -> str:\n"
                        f"    return {constant!r}\n"
                    ),
                    predict=lambda inputs, context, c=constant: c,
                    complexity=1.4 + min(1.0, len(constant) / 24.0),
                )
            )
    for key in curriculum.features.get("string_context_keys", [])[:6]:
        key_lit = repr(str(key))
        sketches.extend(
            [
                AnalyzerSketch(
                    name=_safe_name(f"nova_context_{key}"),
                    description=f"NOVA candidate: return context string {key}",
                    code=(
                        "def rule(current: str, context: dict) -> str:\n"
                        f"    return str(context.get({key_lit}, ''))\n"
                    ),
                    predict=lambda inputs, context, k=str(key): str(context.get(k, "")),
                    complexity=1.25,
                ),
                AnalyzerSketch(
                    name=_safe_name(f"nova_current_plus_context_{key}"),
                    description=f"NOVA candidate: append context string {key}",
                    code=(
                        "def rule(current: str, context: dict) -> str:\n"
                        f"    return current + str(context.get({key_lit}, ''))\n"
                    ),
                    predict=lambda inputs, context, k=str(key): str(inputs) + str(context.get(k, "")),
                    complexity=1.6,
                ),
            ]
        )
    return sketches


def _numeric_law_sketches(curriculum: OracleCurriculum) -> list[AnalyzerSketch]:
    tasks = curriculum.tasks
    keys = [
        str(k)
        for k in curriculum.features.get("numeric_input_keys", [])
    ][:8]
    sketches: list[AnalyzerSketch] = [
        AnalyzerSketch(
            name="nova_numeric_constant",
            description="NOVA numeric candidate: constant law",
            code="def law(inputs: dict) -> float:\n    return 1.0\n",
            predict=lambda inputs, context: 1.0,
            complexity=1.0,
        )
    ]
    for key in keys:
        sketches.append(_numeric_single_key_sketch(key))
    power = _fit_power_law(tasks, keys)
    if power is not None:
        constant, powers = power
        code = _power_law_source(constant, powers)
        name_bits = "_".join(f"{k}_{_power_token(v)}" for k, v in powers.items())
        sketches.append(
            AnalyzerSketch(
                name=_safe_name(f"nova_power_law_{name_bits}")[:96],
                description="NOVA fit a multiplicative power-law prior on local oracle tasks",
                code=code,
                predict=lambda inputs, context, c=constant, p=dict(powers): _predict_power_law(inputs, c, p),
                complexity=1.5 + 0.45 * len(powers),
            )
        )
    return sketches


def _table_analyzer_sketches(
    curriculum: OracleCurriculum,
    *,
    question: str = "",
    column_descriptions: Mapping[str, str] | None = None,
    domain_context: str = "",
) -> list[AnalyzerSketch]:
    descriptions = dict(column_descriptions or {})
    first_df = curriculum.tasks[0].inputs if curriculum.tasks else None
    columns = [str(c) for c in getattr(first_df, "columns", [])]
    numeric_columns = _numeric_columns(first_df, columns)
    relevant = _rank_relevant_columns(
        columns,
        question=question,
        descriptions=descriptions,
        domain_context=domain_context,
    )
    relevant_numeric = [c for c in relevant if c in numeric_columns]
    if len(relevant_numeric) < 2:
        relevant_numeric = numeric_columns[:8]
    else:
        relevant_numeric = relevant_numeric[:8]
    sketches = [
        AnalyzerSketch(
            name="nova_table_shape",
            description="NOVA table candidate: dataframe shape analyzer",
            code=(
                "def analyze(df) -> dict:\n"
                "    return {'evidence': f'rows={len(df)}, columns={len(df.columns)}', "
                "'statistic': float(len(df))}\n"
            ),
            predict=lambda inputs, context: {
                "evidence": f"rows={len(inputs)}, columns={len(inputs.columns)}",
                "statistic": float(len(inputs)),
            },
            complexity=1.0,
        )
    ]
    if len(relevant_numeric) >= 2:
        sketches.append(_table_pair_association_sketch(relevant_numeric, question))
    if len(relevant_numeric) >= 3:
        sketches.append(_table_mediated_chain_sketch(relevant_numeric, question))
    return sketches


def _table_pair_association_sketch(cols: Sequence[str], question: str) -> AnalyzerSketch:
    cols_lit = repr(list(cols))
    code = f'''def analyze(df) -> dict:
    cols = [c for c in {cols_lit} if c in df.columns]
    if len(cols) < 2:
        return {{"evidence": "No question-relevant numeric pair was available.", "statistic": 0.0, "variables": []}}
    data = df[cols].apply(pd.to_numeric, errors="coerce")
    best = None
    best_abs = -1.0
    best_r = 0.0
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            try:
                r = float(data[c1].corr(data[c2]))
            except Exception:
                continue
            if r == r and abs(r) > best_abs:
                best = (c1, c2)
                best_r = r
                best_abs = abs(r)
    if best is None:
        return {{"evidence": "No stable question-relevant association was found.", "statistic": 0.0, "variables": cols[:2]}}
    direction = "positive" if best_r >= 0 else "negative"
    evidence = (
        f"Question-relevant variables {{best[0]}} and {{best[1]}} show the strongest "
        f"{{direction}} association among the selected numeric columns (r={{best_r:.3f}})."
    )
    return {{
        "evidence": evidence,
        "statistic": float(abs(best_r)),
        "variables": [best[0], best[1]],
        "cause": best[0],
        "outcome": best[1],
        "relation": direction + " association",
    }}
'''
    return AnalyzerSketch(
        name="nova_table_pair_association",
        description=(
            "NOVA table candidate: choose the strongest association among "
            "question-relevant numeric variables"
        ),
        code=code,
        predict=lambda inputs, context: {"evidence": f"association over {list(cols)[:3]}", "statistic": 1.0},
        complexity=2.4,
    )


def _table_mediated_chain_sketch(cols: Sequence[str], question: str) -> AnalyzerSketch:
    cols_lit = repr(list(cols))
    code = f'''def analyze(df) -> dict:
    cols = [c for c in {cols_lit} if c in df.columns]
    if len(cols) < 3:
        return {{"evidence": "No three-variable chain was available.", "statistic": 0.0, "variables": cols}}
    data = df[cols].apply(pd.to_numeric, errors="coerce")
    best = None
    best_score = -1.0
    for cause in cols:
        for mediator in cols:
            if mediator == cause:
                continue
            for outcome in cols:
                if outcome == cause or outcome == mediator:
                    continue
                try:
                    r_cm = float(data[cause].corr(data[mediator]))
                    r_mo = float(data[mediator].corr(data[outcome]))
                    r_co = float(data[cause].corr(data[outcome]))
                except Exception:
                    continue
                if r_cm != r_cm or r_mo != r_mo:
                    continue
                score = min(abs(r_cm), abs(r_mo)) + 0.25 * abs(r_co if r_co == r_co else 0.0)
                if score > best_score:
                    best = (cause, mediator, outcome, r_cm, r_mo, r_co if r_co == r_co else 0.0)
                    best_score = score
    if best is None:
        return {{"evidence": "No robust mediated chain was found.", "statistic": 0.0, "variables": cols[:3]}}
    d1 = "positive" if best[3] >= 0 else "negative"
    d2 = "positive" if best[4] >= 0 else "negative"
    evidence = (
        f"Question-relevant chain {{best[0]}} -> {{best[1]}} -> {{best[2]}} is supported by "
        f"a {{d1}} cause-mediator association (r={{best[3]:.3f}}) and a {{d2}} "
        f"mediator-outcome association (r={{best[4]:.3f}}); direct cause-outcome r={{best[5]:.3f}}."
    )
    return {{
        "evidence": evidence,
        "statistic": float(best_score),
        "variables": [best[0], best[1], best[2]],
        "cause": best[0],
        "mediator": best[1],
        "outcome": best[2],
        "relation": "mediated association",
    }}
'''
    return AnalyzerSketch(
        name="nova_table_mediated_chain",
        description=(
            "NOVA table candidate: search a question-relevant cause-mediator-outcome "
            "chain using local correlations"
        ),
        code=code,
        predict=lambda inputs, context: {"evidence": f"mediated chain over {list(cols)[:3]}", "statistic": 1.0},
        complexity=3.0,
    )


def _numeric_single_key_sketch(key: str) -> AnalyzerSketch:
    key_lit = repr(key)
    return AnalyzerSketch(
        name=_safe_name(f"nova_numeric_{key}"),
        description=f"NOVA numeric candidate: use variable {key}",
        code=(
            "def law(inputs: dict) -> float:\n"
            f"    return float(inputs.get({key_lit}, 0.0))\n"
        ),
        predict=lambda inputs, context, k=key: float(inputs.get(k, 0.0)),
        complexity=1.2,
    )


def _numeric_columns(df: Any, columns: Sequence[str]) -> list[str]:
    numeric = []
    if df is None:
        return numeric
    for col in columns:
        try:
            series = df[col]
            converted = series
            if str(getattr(series, "dtype", "")).lower() == "object":
                converted = series.astype(str).str.replace(",", ".", regex=False)
            values = converted.astype(float)
            if float(values.notna().mean()) >= 0.6:
                numeric.append(str(col))
        except Exception:
            continue
    return numeric


def _rank_relevant_columns(
    columns: Sequence[str],
    *,
    question: str,
    descriptions: Mapping[str, str],
    domain_context: str = "",
) -> list[str]:
    q_tokens = _tokens(f"{question} {domain_context}")
    scored: list[tuple[float, str]] = []
    for col in columns:
        text = f"{col} {descriptions.get(str(col), '')}".lower()
        toks = _tokens(text)
        score = 0.0
        score += 4.0 * len(q_tokens & toks)
        for token in q_tokens:
            if token and token in text:
                score += 0.7
        if _is_axis_like(str(col)):
            score -= 1.0
        scored.append((score, str(col)))
    scored.sort(key=lambda item: (item[0], -len(item[1])), reverse=True)
    ranked = [col for score, col in scored if score > 0]
    return ranked or [str(c) for c in columns]


def _tokens(text: str) -> set[str]:
    import re

    stop = {
        "the", "and", "for", "with", "from", "into", "between", "among", "which",
        "what", "when", "where", "does", "did", "were", "was", "are", "how",
        "there", "this", "that", "have", "has", "had", "over", "under", "after",
        "before", "during", "in", "of", "to", "a", "an", "on", "by", "as",
        "using", "use", "used", "table", "data", "dataset", "variable",
        "variables", "column", "columns", "row", "rows",
    }
    return {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_/-]{2,}", str(text).lower())
        if token not in stop
    }


def _is_axis_like(column: str) -> bool:
    import re

    low = column.lower()
    return (
        low in {"year", "date", "time", "period", "country", "entity", "id", "index"}
        or bool(re.fullmatch(r"\d{4}(?:\s*\[yr\d{4}\])?", low))
        or low.endswith("_id")
    )


def _common_suffix_after_input(tasks: Sequence[Any]) -> str | None:
    suffixes = []
    for task in tasks:
        current = str(task.inputs)
        target = str(task.target)
        if not target.startswith(current):
            return None
        suffixes.append(target[len(current):])
    if not suffixes:
        return None
    first = suffixes[0]
    return first if all(s == first for s in suffixes) else None


def _common_prefix_before_input(tasks: Sequence[Any]) -> str | None:
    prefixes = []
    for task in tasks:
        current = str(task.inputs)
        target = str(task.target)
        if not target.endswith(current):
            return None
        prefixes.append(target[: len(target) - len(current)])
    if not prefixes:
        return None
    first = prefixes[0]
    return first if all(p == first for p in prefixes) else None


def _constant_target(tasks: Sequence[Any]) -> str | None:
    if not tasks:
        return None
    first = str(tasks[0].target)
    return first if all(str(t.target) == first for t in tasks) else None


def _fit_power_law(
    tasks: Sequence[Any],
    keys: Sequence[str],
) -> tuple[float, dict[str, float]] | None:
    if not keys:
        return None
    rows: list[list[float]] = []
    ys: list[float] = []
    for task in tasks:
        if not isinstance(task.inputs, Mapping):
            continue
        try:
            target = float(task.target)
        except Exception:
            continue
        if target <= 0:
            continue
        row = [1.0]
        ok = True
        for key in keys:
            try:
                value = float(task.inputs.get(key, 0.0))
            except Exception:
                ok = False
                break
            if value <= 0:
                ok = False
                break
            row.append(math.log(value))
        if not ok:
            continue
        rows.append(row)
        ys.append(math.log(target))
    if len(rows) < len(keys) + 1:
        return None
    coeffs = _least_squares(rows, ys)
    if coeffs is None:
        return None
    constant = math.exp(coeffs[0])
    powers = {
        key: _snap_power(coeffs[i + 1])
        for i, key in enumerate(keys)
        if abs(coeffs[i + 1]) > 1e-5
    }
    if not powers:
        return None
    return constant, powers


def _predict_power_law(inputs: Mapping[str, Any], constant: float, powers: Mapping[str, float]) -> float:
    value = float(constant)
    for key, power in powers.items():
        base = float(inputs.get(key, 0.0))
        if base <= 0:
            return 0.0
        value *= base ** float(power)
    return value


def _power_law_source(constant: float, powers: Mapping[str, float]) -> str:
    lines = [
        "def law(inputs: dict) -> float:",
        f"    value = {float(constant)!r}",
    ]
    for key, power in powers.items():
        lines.append(f"    base = float(inputs.get({key!r}, 0.0))")
        lines.append("    if base <= 0:")
        lines.append("        return 0.0")
        lines.append(f"    value *= base ** {float(power)!r}")
    lines.append("    return value")
    return "\n".join(lines) + "\n"


def _least_squares(rows: list[list[float]], ys: list[float]) -> list[float] | None:
    n = len(rows[0])
    ata = [[0.0 for _ in range(n)] for _ in range(n)]
    aty = [0.0 for _ in range(n)]
    for row, y in zip(rows, ys):
        for i in range(n):
            aty[i] += row[i] * y
            for j in range(n):
                ata[i][j] += row[i] * row[j]
    return _solve_linear(ata, aty)


def _solve_linear(a: list[list[float]], b: list[float]) -> list[float] | None:
    n = len(b)
    aug = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-10:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        div = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= div
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            for j in range(col, n + 1):
                aug[r][j] -= factor * aug[col][j]
    return [aug[i][n] for i in range(n)]


def _snap_power(value: float) -> float:
    candidates = [-4, -3, -2.5, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 2.5, 3, 4]
    return float(min(candidates, key=lambda c: abs(c - value)))


def _power_token(value: float) -> str:
    text = str(value).replace("-", "neg").replace(".", "p")
    return text


def _safe_name(value: str) -> str:
    out = []
    for ch in value.lower():
        out.append(ch if ch.isalnum() else "_")
    name = "".join(out).strip("_")
    while "__" in name:
        name = name.replace("__", "_")
    return name or "nova_candidate"
