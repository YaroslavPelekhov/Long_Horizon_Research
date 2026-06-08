"""Zero-shot grammar synthesis for DiscoveryBench measurement operators.

Current db_cpi.py has HAND-WRITTEN measurement operators:
    schema_profile, relevant_columns, correlation, group_difference, time_extrema, ...

This module replaces them with LLM-synthesized operators generated on-the-fly
from the task description and dataset schema. No human writes any analysis function
for a specific task.

Pipeline:
    task_description + dataset_schema + sample_rows
        → LLM proposes N pandas analysis functions
        → sandbox validates each on sample_rows
        → returns list of (name, fn, description)

The synthesized operators are then used inside the existing DB-CPI rendering
pipeline to generate HMS-compatible sub-hypotheses.

Key difference from db_cpi.py:
    db_cpi.py:     fixed operator library → run on any task
    this module:   task-specific operator library → synthesized from schema
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from mars.agents.base import call_llm, make_openai_client


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DBOperator:
    name: str
    description: str
    fn: Callable
    complexity: float = 1.0
    tags: tuple[str, ...] = ("synthesized",)


@dataclass
class DBSynthesisResult:
    proposed: int
    valid: int
    operators: list[DBOperator]
    errors: list[str]
    model: str
    wall_time_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposed": self.proposed,
            "valid": self.valid,
            "model": self.model,
            "wall_time_s": self.wall_time_s,
            "errors": self.errors[:10],
            "operators": [
                {"name": op.name, "description": op.description, "complexity": op.complexity}
                for op in self.operators
            ],
        }


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------

_SAFE_DB_BUILTINS: dict[str, Any] = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "enumerate": enumerate, "float": float, "int": int, "isinstance": isinstance,
    "len": len, "list": list, "max": max, "min": min, "print": print,
    "range": range, "round": round, "set": set, "sorted": sorted,
    "str": str, "sum": sum, "tuple": tuple, "zip": zip,
    "True": True, "False": False, "None": None,
}


def _build_db_namespace(df: Any) -> dict[str, Any]:
    """Build execution namespace with pandas and numpy available."""
    import pandas as pd  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    ns: dict[str, Any] = {"__builtins__": _SAFE_DB_BUILTINS}
    ns["pd"] = pd
    ns["np"] = np
    ns["df"] = df
    return ns


def _sandbox_db_compile(code: str, sample_df: Any) -> tuple[bool, str, Any]:
    """Compile and smoke-test a proposed pandas analysis function."""
    import ast  # noqa: PLC0415
    try:
        ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}", None

    ns = _build_db_namespace(sample_df)
    try:
        exec(compile(code, "<db_synth>", "exec"), ns)
    except Exception as e:
        return False, f"ExecError: {e}", None

    fn = next(
        (v for k, v in ns.items() if callable(v) and not isinstance(v, type) and not k.startswith("_")),
        None,
    )
    if fn is None:
        return False, "no callable found", None

    # Smoke test
    try:
        result = fn(sample_df)
    except Exception as e:
        return False, f"RuntimeError: {e}", None

    if not isinstance(result, (str, dict)):
        return False, f"expected str or dict, got {type(result).__name__}", None

    return True, "", fn


# ---------------------------------------------------------------------------
# Schema extraction
# ---------------------------------------------------------------------------

def extract_schema_description(
    df: Any,
    max_rows: int = 3,
    column_descriptions: dict[str, str] | None = None,
) -> str:
    """Convert a DataFrame into a concise schema description for the LLM.

    column_descriptions: optional dict {col_name: human_readable_description}
    from the benchmark metadata (e.g. {"ZDolch": "Daggers", "ZBeil": "Axes & Celts"}).
    Including these dramatically improves zero-shot operator synthesis on datasets
    with non-obvious column names.
    """
    lines = ["Dataset schema:"]
    for col in df.columns:
        dtype = str(df[col].dtype)
        sample_vals = df[col].dropna().head(3).tolist()
        desc = ""
        if column_descriptions and col in column_descriptions:
            desc = f" — {column_descriptions[col]}"
        lines.append(f"  '{col}' ({dtype}){desc}: e.g. {sample_vals}")
    lines.append(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
    sample = df.head(max_rows).to_dict(orient="records")
    lines.append(f"\nSample rows:\n{json.dumps(sample, indent=2, default=str)[:1500]}")
    return "\n".join(lines)


def extract_column_descriptions(ds_meta: dict) -> dict[str, str]:
    """Extract column name → description from DiscoveryBench dataset metadata."""
    result: dict[str, str] = {}
    for col_info in ds_meta.get("columns", {}).get("raw", []):
        name = col_info.get("name", "")
        desc = col_info.get("description", "")
        if name and desc:
            result[name] = desc
    return result


# ---------------------------------------------------------------------------
# Core synthesis
# ---------------------------------------------------------------------------

_DB_INTERFACE = """
You are a data analysis synthesizer. Your task is to propose Python functions
that extract statistical evidence from a pandas DataFrame to answer a research question.

Function signature (MUST match exactly):
  def analyze_NAME(df: pd.DataFrame) -> dict:
      # Returns: {
      #   'evidence': str,       # plain English description of what was found
      #   'statistic': float,    # the key numeric finding (or None)
      #   'method': str,         # what statistical operation was performed
      # }

Available: pd (pandas), np (numpy), df (the DataFrame)
Use only standard pandas/numpy operations. No external imports.
"""


def synthesize_db_grammar(
    task_description: str,
    df: Any,
    *,
    column_descriptions: dict[str, str] | None = None,
    model: str = "openai/gpt-4o-mini",
    n_proposals: int = 10,
    temperature: float = 0.7,
) -> DBSynthesisResult:
    """Zero-shot synthesis of measurement operators for a specific DB task.

    Given only the task description and dataset schema, proposes analysis
    functions without any task-specific hand-written operators.
    """
    t0 = time.time()
    schema_desc = extract_schema_description(df, column_descriptions=column_descriptions)

    prompt = f"""{_DB_INTERFACE}

RESEARCH QUESTION:
{task_description}

{schema_desc}

Propose {n_proposals} diverse Python analysis functions.
Cover: correlations, group comparisons, trend analysis, regression coefficients,
frequency counts, temporal patterns, categorical breakdowns, effect sizes.
Each function should test a DIFFERENT aspect of the research question.

Return ONLY this JSON:
{{
  "operators": [
    {{
      "name": "analyze_short_name",
      "description": "one-line description",
      "complexity": <1-5>,
      "code": "def analyze_short_name(df):\\n    ..."
    }}
  ]
}}"""

    client = make_openai_client()
    raw = call_llm(
        client,
        model=model,
        system=(
            "You propose pandas analysis functions to answer research questions. "
            "Return only valid JSON with the operators array."
        ),
        user=prompt,
        max_tokens=4500,
        temperature=temperature,
    )

    proposals = _parse_db_proposals(raw)
    operators: list[DBOperator] = []
    errors: list[str] = []

    for p in proposals:
        if not isinstance(p, dict):
            continue
        code = p.get("code", "").strip()
        if not code:
            errors.append(f"{p.get('name','?')}: missing code")
            continue
        ok, err, fn = _sandbox_db_compile(code, df)
        if not ok:
            errors.append(f"{p.get('name','?')}: {err}")
            continue
        op = DBOperator(
            name=p.get("name", f"analyze_{len(operators)}"),
            description=p.get("description", ""),
            fn=fn,
            complexity=float(p.get("complexity", 2.0)),
        )
        operators.append(op)

    return DBSynthesisResult(
        proposed=len(proposals),
        valid=len(operators),
        operators=operators,
        errors=errors,
        model=model,
        wall_time_s=time.time() - t0,
    )


def _parse_db_proposals(raw: str) -> list[dict]:
    if not raw:
        return []
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).lstrip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            for key in ("operators", "primitives", "functions"):
                if isinstance(obj.get(key), list):
                    return obj[key]
        if isinstance(obj, list):
            return obj
    except Exception:
        pass
    m = re.search(r'\[.*\]', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    m2 = re.search(r'\{.*\}', text, re.DOTALL)
    if m2:
        try:
            obj = json.loads(m2.group())
            if isinstance(obj, dict):
                for key in ("operators", "primitives", "functions"):
                    if isinstance(obj.get(key), list):
                        return obj[key]
        except Exception:
            pass
    return []


# ---------------------------------------------------------------------------
# Evidence collection
# ---------------------------------------------------------------------------

def run_operators(
    operators: list[DBOperator],
    df: Any,
) -> list[dict[str, Any]]:
    """Execute all synthesized operators and collect evidence."""
    results = []
    for op in operators:
        try:
            out = op.fn(df)
            if isinstance(out, str):
                out = {"evidence": out, "statistic": None, "method": op.name}
            results.append({
                "operator": op.name,
                "description": op.description,
                "evidence": out.get("evidence", str(out)),
                "statistic": out.get("statistic"),
                "method": out.get("method", op.name),
                "error": None,
            })
        except Exception as e:
            results.append({
                "operator": op.name,
                "description": op.description,
                "evidence": None,
                "statistic": None,
                "method": op.name,
                "error": str(e),
            })
    return [r for r in results if r["evidence"]]
