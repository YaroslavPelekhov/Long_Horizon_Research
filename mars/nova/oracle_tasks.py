"""Local oracle tasks for MARS-NOVA.

NOVA's first move is to turn the current instance into a small supervised
curriculum whose labels come from observed/executable structure, not from LLM
belief.  This module is intentionally benchmark-agnostic and uses duck typing
for observations so it can be called from UniversalCPI without creating an
import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class OracleTask:
    """One locally supervised micro-task.

    `target` is known by construction from the current interface: an observed
    transition, a held-out row, a verifier output, or a deterministic statistic.
    """

    id: str
    inputs: Any
    target: Any
    context: Mapping[str, Any] = field(default_factory=dict)
    contract: Mapping[str, Any] = field(default_factory=dict)
    source: str = "observed"


@dataclass(frozen=True)
class OracleCurriculum:
    """A compact collection of local oracle tasks and interface facts."""

    tasks: tuple[OracleTask, ...]
    features: Mapping[str, Any]

    def target_kind(self) -> str:
        return str(self.features.get("target_kind", "unknown"))


def build_oracle_curriculum(
    observations: Sequence[Any],
    *,
    interface_name: str = "unknown",
    max_tasks: int = 64,
) -> OracleCurriculum:
    """Build local oracle tasks from current observations only.

    The default curriculum uses observed input/target pairs as supervised
    micro-tasks and records generic interface features.  Benchmark adapters can
    later add richer oracle tasks, but this core path has no benchmark names or
    answer templates.
    """

    tasks: list[OracleTask] = []
    input_types: set[str] = set()
    target_types: set[str] = set()
    context_keys: list[str] = []
    for i, obs in enumerate(list(observations)[:max_tasks]):
        inputs = getattr(obs, "inputs", None)
        target = getattr(obs, "target", None)
        context = dict(getattr(obs, "context", {}) or {})
        input_types.add(type(inputs).__name__)
        target_types.add(type(target).__name__)
        for key in context:
            if key.startswith("__"):
                continue
            if key not in context_keys:
                context_keys.append(key)
        tasks.append(
            OracleTask(
                id=f"{interface_name}_observed_{i}",
                inputs=inputs,
                target=target,
                context=context,
                contract={
                    "input_type": type(inputs).__name__,
                    "target_type": type(target).__name__,
                },
                source="observed_transition",
            )
        )

    features = {
        "interface_name": interface_name,
        "n_tasks": len(tasks),
        "input_types": sorted(input_types),
        "target_types": sorted(target_types),
        "target_kind": _target_kind([t.target for t in tasks]),
        "context_keys": context_keys[:32],
    }
    if tasks and all(isinstance(t.inputs, dict) for t in tasks):
        features["numeric_input_keys"] = _numeric_input_keys(tasks)
    if tasks and all(isinstance(t.inputs, str) for t in tasks):
        features["string_context_keys"] = _string_context_keys(tasks)
        features["int_context_keys"] = _int_context_keys(tasks)
    return OracleCurriculum(tasks=tuple(tasks), features=features)


def _target_kind(targets: Sequence[Any]) -> str:
    if not targets:
        return "unknown"
    if all(isinstance(t, str) for t in targets):
        return "string"
    if all(isinstance(t, (int, float)) and not isinstance(t, bool) for t in targets):
        return "number"
    if all(isinstance(t, Mapping) for t in targets):
        return "mapping"
    return type(targets[0]).__name__


def _numeric_input_keys(tasks: Sequence[OracleTask]) -> list[str]:
    keys: list[str] = []
    for task in tasks:
        if not isinstance(task.inputs, Mapping):
            continue
        for key, value in task.inputs.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                key_s = str(key)
                if key_s not in keys:
                    keys.append(key_s)
    return keys[:32]


def _string_context_keys(tasks: Sequence[OracleTask]) -> list[str]:
    keys: list[str] = []
    for task in tasks:
        for key, value in task.context.items():
            if key.startswith("__"):
                continue
            if isinstance(value, str) and key not in keys:
                keys.append(key)
    return keys[:32]


def _int_context_keys(tasks: Sequence[OracleTask]) -> list[str]:
    keys: list[str] = []
    for task in tasks:
        for key, value in task.context.items():
            if key.startswith("__"):
                continue
            if isinstance(value, int) and not isinstance(value, bool) and key not in keys:
                keys.append(key)
    return keys[:32]
