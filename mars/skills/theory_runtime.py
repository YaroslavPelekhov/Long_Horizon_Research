"""Runtime helpers for feeding BenchmarkTheory into MARS episodes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .benchmark_theory import BenchmarkTheory


def benchmark_theory_from_dict(data: dict[str, Any]) -> BenchmarkTheory:
    """Rehydrate a BenchmarkTheory from the JSON emitted by the compiler."""

    from .benchmark_theory import (  # local import avoids a long public surface
        ActionSpec,
        CheckSpec,
        MetricSpec,
        ResidualSpec,
        TheorySkillSchema,
    )

    return BenchmarkTheory(
        benchmark_name=data["benchmark_name"],
        theory_id=data["theory_id"],
        task_object=data["task_object"],
        observable_inputs=tuple(data.get("observable_inputs", ())),
        target_outputs=tuple(data.get("target_outputs", ())),
        actions=tuple(ActionSpec(**x) for x in data.get("actions", ())),
        metrics=tuple(MetricSpec(**x) for x in data.get("metrics", ())),
        checks=tuple(CheckSpec(**x) for x in data.get("checks", ())),
        residuals=tuple(ResidualSpec(**x) for x in data.get("residuals", ())),
        skill_schemas=tuple(TheorySkillSchema(**x) for x in data.get("skill_schemas", ())),
        compression_objective=data.get("compression_objective", ""),
        evidence={k: tuple(v) for k, v in (data.get("evidence") or {}).items()},
        confidence=float(data.get("confidence", 0.0)),
        notes=tuple(data.get("notes", ())),
    )


def load_compiled_theory(path: str | Path, benchmark_name: str) -> BenchmarkTheory:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    theories = payload.get("theories", [])
    for theory in theories:
        if theory.get("benchmark_name") == benchmark_name:
            return benchmark_theory_from_dict(theory)
    raise KeyError(f"theory not found for benchmark: {benchmark_name}")


def render_theory_context(theory: BenchmarkTheory, *, max_chars: int = 2400) -> str:
    """Render a compact operational contract for a Generator prompt."""

    actions = "; ".join(
        f"{a.name}: {a.description} verifier={a.verifier_signal}"
        for a in theory.actions
    )
    metrics = "; ".join(
        f"{m.name}({m.score_range}) artifact={m.target_artifact} evaluator={m.evaluator}"
        for m in theory.metrics
    )
    residuals = "; ".join(
        f"{r.name}: trigger={r.trigger} repair={r.repair_operator}"
        for r in theory.residuals
    )
    skills = "; ".join(
        f"{s.name}: detect={s.detector}; verify={s.verifier}; repair={s.repair}"
        for s in theory.skill_schemas
    )
    text = (
        "SELF-INDUCED BENCHMARK THEORY\n"
        f"benchmark={theory.benchmark_name} theory_id={theory.theory_id} confidence={theory.confidence:.3f}\n"
        f"task_object={theory.task_object}\n"
        f"observable_inputs={', '.join(theory.observable_inputs)}\n"
        f"target_outputs={', '.join(theory.target_outputs)}\n"
        f"metrics={metrics}\n"
        f"actions={actions}\n"
        f"residual_classes={residuals}\n"
        f"skill_schemas={skills}\n"
        f"compression_objective={theory.compression_objective}\n"
        "OPERATING RULE: before final submission, identify which residual class "
        "your current answer risks, run the matching verifier/check if available, "
        "and repair through the named skill schema."
    )
    if len(text) > max_chars:
        return text[: max_chars - 40] + "\n[theory context truncated]"
    return text
