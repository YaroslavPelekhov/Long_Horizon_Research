"""Executable baselines generated from compiled task contracts.

These baselines are not benchmark-specific answers.  They are small programs
constructed from file contracts, output contracts, and evaluator metric hints.
High-confidence baselines can be executed before LLM candidates; lower
confidence baselines are useful as scaffold candidates or repair anchors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from .task_contract import TaskContract


@dataclass(frozen=True)
class ContractBaseline:
    name: str
    code: str
    auto_accept: bool
    rationale: str


def generate_contract_baselines(contract: TaskContract, *, task_text: str = "") -> list[ContractBaseline]:
    baselines: list[ContractBaseline] = []
    npy = _npy_mapping_baseline(contract, task_text=task_text)
    if npy:
        baselines.append(npy)
    csv = _csv_metric_baseline(contract)
    if csv:
        baselines.append(csv)
    return baselines


def generate_auto_baseline_program(contract: TaskContract, *, task_text: str = "") -> str:
    for baseline in generate_contract_baselines(contract, task_text=task_text):
        if baseline.auto_accept:
            return baseline.code
    return ""


def generate_dataframe_analyzer_baselines(
    df: Any,
    *,
    question: str = "",
    column_descriptions: dict[str, str] | None = None,
    max_columns: int = 28,
) -> list[ContractBaseline]:
    """Generate executable `analyze(df)` programs from a DataFrame interface.

    The generated programs use only observable schema, question text, and
    column descriptions.  They are meant to be scored by the CPI engine like any
    other seed, then rendered into a hypothesis by the adapter.
    """
    cols = [str(c) for c in getattr(df, "columns", [])]
    if not cols:
        return []
    descriptions = column_descriptions or {}
    numeric_cols = _numeric_columns(df, cols)
    if not numeric_cols:
        return []
    relevant = _rank_relevant_columns(cols, question=question, descriptions=descriptions)
    relevant_numeric = [c for c in relevant if c in numeric_cols]
    if len(relevant_numeric) < 2:
        relevant_numeric = numeric_cols[:max_columns]
    else:
        relevant_numeric = relevant_numeric[:max_columns]
    baselines = [
        ContractBaseline(
            name="df_relevant_correlation_probe",
            code=_dataframe_correlation_probe(relevant_numeric),
            auto_accept=False,
            rationale="question-grounded numeric columns support pairwise association measurement",
        ),
        ContractBaseline(
            name="df_temporal_or_axis_extrema_probe",
            code=_dataframe_extrema_probe(relevant_numeric, _axis_like_columns(cols)),
            auto_accept=False,
            rationale="axis-like columns plus numeric variables support extrema/onset evidence",
        ),
    ]
    if len(relevant_numeric) >= 3:
        baselines.append(
            ContractBaseline(
                name="df_multivariate_stability_probe",
                code=_dataframe_stability_probe(relevant_numeric),
                auto_accept=False,
                rationale="multiple relevant numeric columns support split-stability evidence",
            )
        )
    return baselines


def _npy_mapping_baseline(contract: TaskContract, *, task_text: str) -> ContractBaseline | None:
    if contract.output_kind != "npy" or "spearmanr" not in contract.evaluator_metrics:
        return None
    npy_files = [f.path for f in contract.dataset_files if f.kind == "npy"]
    train_files = [p for p in npy_files if "/train/" in p]
    test_files = [p for p in npy_files if "/test/" in p]
    if len(train_files) < 2 or not test_files:
        return None
    source_token, target_token = source_target_tokens(
        {"task_inst": task_text, "output_fname": contract.output_path}
    )
    source_train = _find_token_path(train_files, source_token) or train_files[0]
    target_train = _find_token_path([p for p in train_files if p != source_train], target_token)
    if not target_train:
        target_train = next((p for p in train_files if p != source_train), "")
    source_test = _find_token_path(test_files, source_token) or test_files[0]
    if not target_train or not source_test:
        return None
    code = f"""import os
import numpy as np
from sklearn.linear_model import Ridge

source_train_path = {source_train!r}
target_train_path = {target_train!r}
source_test_path = {source_test!r}
output_path = {contract.output_path!r}

os.makedirs(os.path.dirname(output_path), exist_ok=True)
source_train = np.load(source_train_path)
target_train = np.load(target_train_path)
source_test = np.load(source_test_path)

def normalize_with_train(a, ref):
    std = float(np.std(ref))
    if std == 0:
        std = 1.0
    return (a - float(np.mean(ref))) / std

def flatten_trials(a):
    if a.ndim == 3:
        return np.transpose(a, (1, 0, 2)).reshape(a.shape[1], a.shape[0] * a.shape[2])
    return a.reshape(a.shape[0], -1)

X = flatten_trials(normalize_with_train(source_train, source_train)).astype("float32")
Y = flatten_trials(normalize_with_train(target_train, target_train)).astype("float32")
T = flatten_trials(normalize_with_train(source_test, source_train)).astype("float32")
n = min(12000, X.shape[0])
idx = np.linspace(0, X.shape[0] - 1, n).astype(int)
model = Ridge(alpha=100.0, fit_intercept=True)
model.fit(X[idx], Y[idx])
pred = model.predict(T).astype("float32")
assert pred.shape[0] == T.shape[0]
np.save(output_path, pred)
"""
    return ContractBaseline(
        name="paired_npy_ridge_mapping",
        code=code,
        auto_accept=True,
        rationale="paired NPY files plus Spearman evaluator imply fast rank-preserving linear mapping",
    )


def _csv_metric_baseline(contract: TaskContract) -> ContractBaseline | None:
    if contract.output_kind != "csv" or not contract.target_candidates or not contract.recommended_features:
        return None
    train = next(
        (
            f for f in contract.dataset_files
            if f.kind == "csv" and ("train" in Path(f.path).name.lower() or "/train/" in f.path)
        ),
        None,
    )
    test = next(
        (
            f for f in contract.dataset_files
            if f.kind == "csv" and ("test" in Path(f.path).name.lower() or "/test/" in f.path)
        ),
        None,
    )
    if not train or not test:
        return None
    target = contract.target_candidates[0]
    id_col = contract.id_columns[0] if contract.id_columns else (
        contract.required_output_columns[0] if contract.required_output_columns else ""
    )
    output_columns = list(contract.required_output_columns) or [id_col, target]
    feature_cols = list(contract.recommended_features)
    metric = "roc_auc_score" if "roc_auc_score" in contract.evaluator_metrics else "generic"
    n_estimators = 300 if metric == "roc_auc_score" else 200
    code = f"""import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

train = pd.read_csv({train.path!r})
test = pd.read_csv({test.path!r})
target_col = {target!r}
id_col = {id_col!r}
feature_cols = {feature_cols!r}
output_columns = {output_columns!r}
output_path = {contract.output_path!r}
os.makedirs(os.path.dirname(output_path), exist_ok=True)
feature_cols = [c for c in feature_cols if c in train.columns and c in test.columns and c != target_col]
X_train = train[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
X_test = test[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
y_train = pd.to_numeric(train[target_col], errors="coerce").fillna(train[target_col].median())
model = RandomForestRegressor(n_estimators={n_estimators}, random_state=0, min_samples_leaf=1)
model.fit(X_train, y_train)
pred = model.predict(X_test)
out = pd.DataFrame()
if id_col and id_col in test.columns:
    out[id_col] = test[id_col]
out[target_col] = pred
for col in output_columns:
    if col not in out.columns:
        out[col] = test[col] if col in test.columns else pred
out = out[output_columns] if output_columns else out
out.to_csv(output_path, index=False)
"""
    return ContractBaseline(
        name="csv_metric_regression_scaffold",
        code=code,
        auto_accept=False,
        rationale="CSV train/test contract with numeric shared features; kept as scaffold unless validated",
    )


def _numeric_columns(df: Any, cols: list[str]) -> list[str]:
    numeric: list[str] = []
    for col in cols:
        try:
            series = df[col]
            converted = series
            if str(getattr(series, "dtype", "")).lower() == "object":
                converted = series.astype(str).str.replace(",", ".", regex=False)
            values = converted.astype(float)
            if values.notna().mean() >= 0.6:
                numeric.append(col)
        except Exception:
            continue
    return numeric


def _rank_relevant_columns(
    cols: list[str],
    *,
    question: str,
    descriptions: dict[str, str],
) -> list[str]:
    q_tokens = _tokens(question)
    scored: list[tuple[float, str]] = []
    for col in cols:
        text = f"{col} {descriptions.get(col, '')}"
        toks = _tokens(text)
        score = len(q_tokens & toks) * 3.0
        for token in q_tokens:
            if token and token in text.lower():
                score += 0.5
        if _is_axis_like(col):
            score -= 1.0
        scored.append((score, col))
    scored.sort(key=lambda x: (x[0], -len(x[1])), reverse=True)
    ranked = [col for score, col in scored if score > 0]
    return ranked or cols


def _tokens(text: str) -> set[str]:
    stop = {
        "the", "and", "for", "with", "from", "into", "between", "among", "which",
        "what", "when", "where", "does", "did", "were", "was", "are", "how",
        "there", "this", "that", "have", "has", "had", "over", "under", "after",
        "before", "during", "in", "of", "to", "a", "an", "on", "by", "as",
    }
    return {
        t for t in re.findall(r"[a-zA-Z][a-zA-Z0-9_/-]{2,}", text.lower())
        if t not in stop
    }


def _axis_like_columns(cols: list[str]) -> list[str]:
    return [c for c in cols if _is_axis_like(c)]


def _is_axis_like(col: str) -> bool:
    low = col.lower()
    return (
        low in {"year", "date", "time", "period", "country", "entity", "id", "index"}
        or bool(re.fullmatch(r"\d{4}(?:\s*\[yr\d{4}\])?", low))
        or low.endswith("_id")
    )


def _dataframe_correlation_probe(cols: list[str]) -> str:
    return f'''def analyze(df):
    cols = {cols!r}
    cols = [c for c in cols if c in df.columns]
    if len(cols) < 2:
        return {{"evidence": "Not enough relevant numeric columns for association probe.", "statistic": 0.0}}
    data = df[cols].apply(pd.to_numeric, errors="coerce")
    corr = data.corr(numeric_only=True).abs()
    best = None
    best_val = -1.0
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            try:
                val = float(corr.loc[c1, c2])
            except Exception:
                continue
            if val == val and val > best_val:
                best_val = val
                best = (c1, c2)
    if best is None:
        return {{"evidence": "No stable pairwise association found.", "statistic": 0.0}}
    direction = "positive" if float(data[best[0]].corr(data[best[1]])) >= 0 else "negative"
    return {{
        "evidence": f"Across the table, {{best[0]}} and {{best[1]}} show the strongest {{direction}} association among relevant numeric variables (|r|={{best_val:.3f}}).",
        "statistic": float(best_val),
    }}
'''


def _dataframe_extrema_probe(cols: list[str], axes: list[str]) -> str:
    return f'''def analyze(df):
    cols = {cols!r}
    axes = {axes!r}
    cols = [c for c in cols if c in df.columns]
    axis = next((a for a in axes if a in df.columns), None)
    if not cols:
        return {{"evidence": "No relevant numeric variable found for extrema probe.", "statistic": 0.0}}
    data = df[cols].apply(pd.to_numeric, errors="coerce")
    best_col = None
    best_score = -1.0
    best_idx = None
    for col in cols:
        s = data[col]
        spread = float(s.max() - s.min()) if s.notna().any() else 0.0
        if spread > best_score:
            best_score = spread
            best_col = col
            try:
                best_idx = int(s.idxmax())
            except Exception:
                best_idx = None
    if best_col is None:
        return {{"evidence": "No extrema evidence found.", "statistic": 0.0}}
    where = ""
    if axis is not None and best_idx is not None:
        try:
            where = f" at {{axis}}={{df.loc[best_idx, axis]}}"
        except Exception:
            where = ""
    value = float(data[best_col].max())
    return {{
        "evidence": f"Variable {{best_col}} has the clearest maximum{{where}} (max={{value:.3g}}, spread={{best_score:.3g}}), suggesting an event or concentration in that context.",
        "statistic": float(best_score),
    }}
'''


def _dataframe_stability_probe(cols: list[str]) -> str:
    return f'''def analyze(df):
    cols = {cols!r}
    cols = [c for c in cols if c in df.columns]
    if len(cols) < 3 or len(df) < 4:
        return {{"evidence": "Not enough data for split-stability probe.", "statistic": 0.0}}
    half = len(df) // 2
    data_a = df.iloc[:half][cols].apply(pd.to_numeric, errors="coerce")
    data_b = df.iloc[half:][cols].apply(pd.to_numeric, errors="coerce")
    best = None
    best_val = -1.0
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            try:
                r1 = abs(float(data_a[c1].corr(data_a[c2])))
                r2 = abs(float(data_b[c1].corr(data_b[c2])))
            except Exception:
                continue
            if r1 == r1 and r2 == r2:
                val = min(r1, r2)
                if val > best_val:
                    best_val = val
                    best = (c1, c2, r1, r2)
    if best is None:
        return {{"evidence": "No split-stable association found.", "statistic": 0.0}}
    return {{
        "evidence": f"Association between {{best[0]}} and {{best[1]}} is stable across table splits (|r|={{best[2]:.3f}} and {{best[3]:.3f}}), making it less likely to be a single-row artifact.",
        "statistic": float(best_val),
    }}
'''


def source_target_tokens(task: dict) -> tuple[str, str]:
    text = f"{task.get('task_inst', '')} {task.get('output_fname', '')}".lower()
    compact = re.sub(r"[^a-z0-9]+", "", text)
    match = re.search(r"sub0?(\d+)to(?:sub)?0?(\d+)", compact)
    if match:
        return f"sub{int(match.group(1)):02d}", f"sub{int(match.group(2)):02d}"
    match = re.search(r"sub\s*0?(\d+).*?to\s*(?:another\s*)?(?:subject\s*)?0?(\d+)", text)
    if match:
        return f"sub{int(match.group(1)):02d}", f"sub{int(match.group(2)):02d}"
    return "", ""


def _find_token_path(paths: list[str], token: str) -> str:
    if not token:
        return ""
    return next((p for p in paths if token.lower() in Path(p).name.lower()), "")
