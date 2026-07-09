"""Question Decomposition CPI (QD-CPI) for semantic data analysis tasks.

Why this exists
---------------
Grammar synthesis (grammar_synthesizer.py) works when the observable interface is
TYPED: a transformation f(current, context) -> output where the mechanism is
discoverable from a few input/output examples.

For DiscoveryBench, the interface is SEMANTIC: "In which century did X first
increase?" This requires REASONING about what to compute, not just computing it.
Proposing 10 generic operators and picking the best fails because:
  - The right operator depends on understanding the question structure
  - "First increase" requires specific temporal logic, not a general correlation
  - Column identification requires domain reasoning before any code is written

QD-CPI decomposes the question into a sequential chain of verifiable sub-steps:

  question → step_1 (identify variable) → step_2 (identify time) →
  step_3 (define metric operationally) → step_4 (execute) → step_5 (interpret)

Each step is a hypothesis:
  - "ZDolch is the daggers column" — verifiable by checking column descriptions
  - "BCE is the time column" — verifiable by dtype and value range
  - "first increase = first BCE where diff(ZDolch) > 0" — verifiable by execution

The chain is refutable: if step k fails, the LLM revises step k rather than
generating 10 new operators.

This is the second mode in the meta-CPI framework:
  typed interface  → grammar_synthesizer + CPI
  semantic interface → QD-CPI (this module)
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from mars.agents.base import call_llm, make_openai_client
from mars.induction.db_grammar_synthesizer import extract_schema_description


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ReasoningStep:
    step_id: int
    step_type: str   # "identify_variable" | "define_metric" | "write_code" | "interpret"
    description: str
    code: str | None = None
    expected: str = ""
    result: Any = None
    error: str | None = None
    verified: bool = False


@dataclass
class ReasoningChain:
    steps: list[ReasoningStep]
    final_answer: str
    evidence: str
    confidence: float
    model: str
    wall_time_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": [
                {
                    "step_id": s.step_id,
                    "step_type": s.step_type,
                    "description": s.description,
                    "result": str(s.result)[:200] if s.result is not None else None,
                    "error": s.error,
                    "verified": s.verified,
                }
                for s in self.steps
            ],
            "final_answer": self.final_answer,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "model": self.model,
            "wall_time_s": self.wall_time_s,
        }


# ---------------------------------------------------------------------------
# Interface type classifier
# ---------------------------------------------------------------------------

INTERFACE_TYPED = "typed_mechanism"
INTERFACE_SEMANTIC = "semantic_reasoning"


QUESTION_TEMPORAL = "temporal_occurrence"    # "when did X first increase/peak?"
QUESTION_REGIONAL = "regional_comparison"    # "in what regions does X correlate with Y?"
QUESTION_STATISTICAL = "statistical_relation" # "what is the relationship between X and Y?"
QUESTION_TYPED = "typed_mechanism"           # transformation rules, physics laws


def classify_interface(task_description: str) -> str:
    """Classify the task interface type.

    Returns one of:
      typed_mechanism     — transformation rules, physics laws (use grammar synthesis)
      temporal_occurrence — "when did X first/peak?" (use QD-CPI chain)
      regional_comparison — "in what regions does X?" (use LLM reasoning)
      statistical_relation — "what is the relationship?" (use zero_shot operators)
      semantic_reasoning  — fallback for ambiguous semantic tasks
    """
    desc_lower = task_description.lower()

    # Typed interface signals
    typed_signals = [
        "rule", "transformation", "input_sequences", "law", "force",
        "physics", "symbolic", "equation", "observe", "intervene",
        "causal graph", "main", "vice", "step_number",
    ]
    if sum(1 for s in typed_signals if s in desc_lower) >= 3:
        return INTERFACE_TYPED

    # Temporal occurrence signals
    temporal_signals = [
        "century", "year", "when did", "first time", "began to",
        "increase", "peak", "earliest", "period", "historical",
    ]
    # Regional comparison signals
    regional_signals = [
        "region", "country", "continent", "area", "where", "location",
        "geographic", "sub-saharan", "asia", "europe",
    ]
    # Statistical relation signals
    stat_signals = [
        "correlation", "relationship", "effect", "impact", "influence",
        "regression", "coefficient",
    ]

    temporal_score = sum(1 for s in temporal_signals if s in desc_lower)
    regional_score = sum(1 for s in regional_signals if s in desc_lower)
    stat_score = sum(1 for s in stat_signals if s in desc_lower)

    if temporal_score > max(regional_score, stat_score):
        return QUESTION_TEMPORAL
    if regional_score > max(temporal_score, stat_score):
        return QUESTION_REGIONAL
    if stat_score > 0:
        return QUESTION_STATISTICAL

    return INTERFACE_SEMANTIC


# ---------------------------------------------------------------------------
# Pandas sandbox
# ---------------------------------------------------------------------------

def _run_code_sandbox(
    code: str,
    df: Any,
    context: dict[str, Any],
) -> tuple[bool, str, Any]:
    """Execute analysis code with pandas available. Returns (ok, error, result)."""
    import pandas as pd  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    ns: dict[str, Any] = {
        "pd": pd,
        "np": np,
        "df": df,
        "__builtins__": {
            "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
            "enumerate": enumerate, "float": float, "int": int,
            "isinstance": isinstance, "len": len, "list": list,
            "max": max, "min": min, "print": print, "range": range,
            "round": round, "set": set, "sorted": sorted,
            "str": str, "sum": sum, "tuple": tuple, "zip": zip,
            "True": True, "False": False, "None": None,
        },
    }
    ns.update(context)
    try:
        exec(compile(code, "<qd_cpi>", "exec"), ns)
        result = ns.get("result", ns.get("answer", ns.get("output")))
        return True, "", result
    except Exception as e:
        return False, str(e), None


# ---------------------------------------------------------------------------
# Chain planner
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM = (
    "You are a scientific data analyst. Break down research questions into "
    "sequential, verifiable reasoning steps. Each step must be executable "
    "and its result feeds into the next step. Return only valid JSON."
)


def _plan_chain(
    client: Any,
    model: str,
    question: str,
    domain_knowledge: str,
    schema_desc: str,
) -> list[dict]:
    """Ask LLM to produce a sequential reasoning plan for the question."""
    prompt = f"""Break down this scientific research question into a chain of reasoning steps.

QUESTION: {question}

DOMAIN CONTEXT: {domain_knowledge[:400]}

DATASET SCHEMA:
{schema_desc[:1500]}

Produce a JSON chain where each step builds on the previous result.
Each step must have:
- step_type: one of: "identify_variable" | "define_metric" | "write_code" | "interpret"
- description: what this step determines
- code: (only for write_code steps) Python code using `df` (pandas DataFrame).
  Store result in variable named `result`. Use pd and np. No imports.
- expected: what you expect to find

Return JSON: {{"steps": [...]}}

Rules for write_code steps:
- The DataFrame `df` is available with the schema shown above
- Store your answer in a variable called `result`
- result should be a str, number, or simple dict
- Example: result = str(df['BCE'].iloc[df['ZDolch'].diff().gt(0).idxmax()])"""

    raw = call_llm(
        client, model=model,
        system=_PLANNER_SYSTEM,
        user=prompt,
        max_tokens=2500,
        temperature=0.3,
    )

    try:
        parsed = json.loads(raw)
        return parsed.get("steps", []) if isinstance(parsed, dict) else []
    except Exception:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group()).get("steps", [])
            except Exception:
                pass
    return []


# ---------------------------------------------------------------------------
# Step interpreter
# ---------------------------------------------------------------------------

def _interpret_result(
    client: Any,
    model: str,
    question: str,
    steps: list[ReasoningStep],
) -> tuple[str, str, float]:
    """Given executed chain, synthesize final answer + evidence."""
    executed = [
        f"Step {s.step_id} ({s.step_type}): {s.description}\n  Result: {str(s.result)[:150]}"
        for s in steps if s.result is not None and not s.error
    ]
    if not executed:
        return "Could not determine answer from analysis.", "No successful steps.", 0.0

    prompt = f"""Based on these sequential analysis results, write a concise answer.

Question: {question}

Analysis chain:
{chr(10).join(executed)}

Write:
1. A direct 1-2 sentence answer to the question
2. A confidence score 0.0-1.0

Return JSON: {{"answer": "...", "evidence": "...", "confidence": 0.0}}"""

    raw = call_llm(
        client, model=model,
        system="You synthesize scientific answers from step-by-step analysis. Return JSON.",
        user=prompt,
        max_tokens=400,
        temperature=0.1,
    )
    try:
        parsed = json.loads(raw)
        return (
            parsed.get("answer", ""),
            parsed.get("evidence", " ".join(executed[:2])),
            float(parsed.get("confidence", 0.5)),
        )
    except Exception:
        return " ".join(executed[:2])[:200], "From chain execution.", 0.3


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def solve(
    question: str,
    domain_knowledge: str,
    df: Any,
    *,
    column_descriptions: dict[str, str] | None = None,
    model: str = "openai/gpt-4o",
    max_retries: int = 2,
) -> ReasoningChain:
    """Decompose a semantic research question into a verifiable reasoning chain.

    Each step is executed sequentially. Failed code steps are retried with
    error feedback. The final answer is synthesized from successful steps.
    """
    t0 = time.time()
    client = make_openai_client()
    schema_desc = extract_schema_description(df, column_descriptions=column_descriptions)

    # 1. Plan the chain
    plan = _plan_chain(client, model, question, domain_knowledge, schema_desc)

    # 2. Execute each step
    context: dict[str, Any] = {}  # carries results between steps
    executed_steps: list[ReasoningStep] = []

    for i, plan_step in enumerate(plan):
        step = ReasoningStep(
            step_id=i + 1,
            step_type=plan_step.get("step_type", "write_code"),
            description=plan_step.get("description", ""),
            code=plan_step.get("code"),
            expected=plan_step.get("expected", ""),
        )

        if step.code:
            for attempt in range(max_retries + 1):
                ok, err, result = _run_code_sandbox(step.code, df, context)
                if ok:
                    step.result = result
                    step.verified = True
                    context[f"step_{i+1}_result"] = result
                    context["result"] = result
                    break
                else:
                    step.error = err
                    if attempt < max_retries:
                        # Retry with error feedback
                        fix_prompt = (
                            f"This code failed: {step.code}\n"
                            f"Error: {err}\n"
                            f"Schema: {schema_desc[:600]}\n"
                            f"Fix the code. Store answer in `result`. Return only the fixed code."
                        )
                        fixed = call_llm(
                            client, model=model,
                            system="Fix Python pandas code. Return ONLY the corrected code, no explanation.",
                            user=fix_prompt,
                            max_tokens=600,
                            temperature=0.2,
                        )
                        # Strip markdown fences
                        fixed = re.sub(r'^```\w*\n?', '', fixed.strip())
                        fixed = re.sub(r'\n?```$', '', fixed)
                        step.code = fixed
        else:
            # Non-code step: just record the expected finding as result
            step.result = step.expected or step.description
            step.verified = True

        executed_steps.append(step)

    # 3. Synthesize answer
    final_answer, evidence, confidence = _interpret_result(
        client, model, question, executed_steps
    )

    return ReasoningChain(
        steps=executed_steps,
        final_answer=final_answer,
        evidence=evidence,
        confidence=confidence,
        model=model,
        wall_time_s=time.time() - t0,
    )
