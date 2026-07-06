"""Benchmark-theory final artifact synthesis.

The Generator explores and asserts local claims.  Benchmark scorers, however,
usually grade one final artifact: a hypothesis, law, program, report, or rule.
This module turns the accumulated episode trace into that artifact under the
compiled BenchmarkTheory contract.
"""

from __future__ import annotations

import json
from typing import Any

from mars.agents.base import call_llm, make_openai_client, parse_json_strict

from .benchmark_theory import BenchmarkTheory
from .theory_runtime import render_theory_context
from ols.core.types import Claim


_SYS = (
    "You are the FinalArtifact synthesizer in a self-improving research system. "
    "Your job is not to brainstorm. Your job is to compress the already observed "
    "evidence into the single artifact that the benchmark metric grades. "
    "Use only the environment brief, active claims, and executed action results. "
    "Do not invent hidden evidence. Reply exactly one JSON object."
)


def _short_json(obj: Any, limit: int = 900) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        s = str(obj)
    return s[:limit]


def _render_claims(claims: list[Claim], *, max_claims: int = 12) -> str:
    if not claims:
        return "(no active claims)"
    ranked = sorted(claims, key=lambda c: c.confidence, reverse=True)[:max_claims]
    return "\n".join(
        f"- conf={c.confidence:.2f} sub_goal={c.sub_goal or '?'}: {c.statement}"
        for c in ranked
    )


def _render_history(history: list[dict], *, max_items: int = 16) -> str:
    if not history:
        return "(no executed action history)"
    lines: list[str] = []
    for h in history[-max_items:]:
        lines.append(
            " | ".join(
                [
                    f"t={h.get('t')}",
                    f"sub_goal={h.get('sub_goal')}",
                    f"actions={h.get('action_results') or h.get('n_actions')}",
                    f"claims={h.get('n_claims')}",
                ]
            )
        )
    return "\n".join(lines)


def synthesize_final_artifact(
    *,
    model: str,
    theory: BenchmarkTheory,
    env_description: str,
    active_claims: list[Claim],
    history: list[dict],
    max_tokens: int = 900,
) -> str:
    """Return a metric-facing final artifact, or ``""`` if synthesis fails."""

    metric_names = ", ".join(m.name for m in theory.metrics) or "benchmark metric"
    target_outputs = ", ".join(theory.target_outputs) or theory.task_object
    usr = (
        f"{render_theory_context(theory, max_chars=2200)}\n\n"
        f"ENVIRONMENT BRIEF:\n{env_description[:3000]}\n\n"
        f"ACTIVE CLAIMS:\n{_render_claims(active_claims)}\n\n"
        f"EXECUTED TRACE SUMMARY:\n{_render_history(history)}\n\n"
        "SYNTHESIS CONTRACT:\n"
        f"- Metric(s): {metric_names}\n"
        f"- Target output type(s): {target_outputs}\n"
        "- Produce exactly the final artifact that should be submitted/scored.\n"
        "- Preserve concrete variable names, constants, code, filenames, and "
        "numeric evidence when present.\n"
        "- If evidence is thin, return the best conservative artifact supported "
        "by the trace; do not add unsupported mechanisms.\n\n"
        "Reply JSON now with fields:\n"
        '{"artifact": "...", "support": ["short evidence item", ...], '
        '"risk": "main residual risk"}'
    )
    try:
        raw = call_llm(make_openai_client(), model, _SYS, usr, max_tokens=max_tokens)
        obj = parse_json_strict(raw)
        if not isinstance(obj, dict):
            return ""
        artifact = str(obj.get("artifact", "")).strip()
        return artifact
    except Exception:
        return ""
