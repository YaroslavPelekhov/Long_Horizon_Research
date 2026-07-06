"""Probe genomes for adversarial theory validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ProbeGenome:
    """A small executable or descriptive pressure applied to theory genomes."""

    name: str
    probe_type: str
    transformation: str
    expected_invariance: str
    cost: float
    bits: Mapping[str, Any]


def build_probe_genomes(
    *,
    signature_hint: str,
    observations: Sequence[Any],
) -> tuple[ProbeGenome, ...]:
    """Create generic probes from the interface, without benchmark answers."""

    if not observations:
        return ()
    hint = str(signature_hint)
    probes: list[ProbeGenome] = [
        ProbeGenome(
            name="heldout_replay",
            probe_type="heldout",
            transformation="split observations into seen and unseen folds",
            expected_invariance="candidate keeps the same explanatory structure across folds",
            cost=1.0,
            bits={"n_observations": len(observations)},
        )
    ]
    if "analyze(df)" in hint:
        df = getattr(observations[-1], "inputs", None)
        if df is not None and hasattr(df, "columns"):
            probes.append(
                ProbeGenome(
                    name="schema_orientation_probe",
                    probe_type="schema",
                    transformation="compare row labels, column labels, and axis-like fields",
                    expected_invariance="variables should bind to the semantic axis, not merely the numeric axis",
                    cost=1.2,
                    bits={
                        "n_columns": len(list(getattr(df, "columns", []))),
                        "n_rows": int(getattr(df, "shape", [0])[0]),
                    },
                )
            )
    if "law(inputs" in hint:
        probes.append(
            ProbeGenome(
                name="scale_perturbation_probe",
                probe_type="perturbation",
                transformation="reason over multiplicative changes in numeric inputs",
                expected_invariance="law should preserve a stable scale relation under input perturbation",
                cost=1.4,
                bits={},
            )
        )
    if "rule(current" in hint:
        probes.append(
            ProbeGenome(
                name="rollout_consistency_probe",
                probe_type="rollout",
                transformation="apply the same transition theory across several steps",
                expected_invariance="one-step rule should remain coherent under repeated use",
                cost=1.3,
                bits={},
            )
        )
    if "predict(parent1" in hint:
        probes.append(
            ProbeGenome(
                name="role_swap_probe",
                probe_type="symmetry",
                transformation="swap structured parent/state roles where the interface is symmetric",
                expected_invariance="prediction should respect observed role symmetry unless data breaks it",
                cost=1.3,
                bits={},
            )
        )
    return tuple(probes)
