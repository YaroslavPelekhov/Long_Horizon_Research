"""Universal epistemic compiler certificates.

This module is the anti-overfitting gate for self-building MARS.

The compiler does not know benchmark answers.  It records whether a candidate
program/operator behaved like a reusable scientific measurement:

    evidence -> representation/probe -> shorter verified hypothesis

The important object is a certificate, not a prompt.  A result is strong when it
improves held-out loss, pays for its own complexity by compression, avoids
benchmark-name leakage, and has a task fingerprint that can be compared across
benchmarks without using benchmark IDs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from typing import Any, Iterable, Mapping, Sequence


_EPS = 1e-12
_BENCHMARK_TOKENS = {
    "newtonbench",
    "newton",
    "discoverybench",
    "discovery",
    "ultrahorizon",
    "ultra",
    "horizon",
    "scienceagentbench",
    "scienceagent",
    "agentbench",
    "sab",
}


@dataclass(frozen=True)
class TaskFingerprint:
    """Benchmark-free observable task signature."""

    signature_hint: str
    interface_family: str
    input_kind: str
    target_kind: str
    context_keys: tuple[str, ...]
    input_keys: tuple[str, ...]
    n_observations: int
    numeric_density: float
    sequence_density: float
    table_density: float
    text_density: float
    observable_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "signature_hint": self.signature_hint,
            "interface_family": self.interface_family,
            "input_kind": self.input_kind,
            "target_kind": self.target_kind,
            "context_keys": list(self.context_keys),
            "input_keys": list(self.input_keys),
            "n_observations": self.n_observations,
            "numeric_density": self.numeric_density,
            "sequence_density": self.sequence_density,
            "table_density": self.table_density,
            "text_density": self.text_density,
            "observable_hash": self.observable_hash,
        }


@dataclass(frozen=True)
class EpistemicCertificate:
    """MDL/Bayes-style certificate for a self-built hypothesis layer."""

    fingerprint: TaskFingerprint
    status: str
    universal_score: float
    compression_gain: float
    stability_score: float
    transfer_support: float
    complexity_penalty: float
    leakage_penalty: float
    best_loss: float
    reference_loss: float
    evidence_count: int
    winner_family: str
    accepted_reasons: tuple[str, ...] = ()
    rejected_reasons: tuple[str, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.status == "accepted"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "accepted": self.accepted,
            "universal_score": round(float(self.universal_score), 6),
            "compression_gain": round(float(self.compression_gain), 6),
            "stability_score": round(float(self.stability_score), 6),
            "transfer_support": round(float(self.transfer_support), 6),
            "complexity_penalty": round(float(self.complexity_penalty), 6),
            "leakage_penalty": round(float(self.leakage_penalty), 6),
            "best_loss": round(float(self.best_loss), 6),
            "reference_loss": round(float(self.reference_loss), 6),
            "evidence_count": int(self.evidence_count),
            "winner_family": self.winner_family,
            "accepted_reasons": list(self.accepted_reasons),
            "rejected_reasons": list(self.rejected_reasons),
            "fingerprint": self.fingerprint.to_dict(),
            "diagnostics": dict(self.diagnostics),
        }


@dataclass
class EpistemicLedger:
    """Cross-task memory for operator-level transfer evidence.

    It is intentionally tiny and serializable.  The ledger stores abstract
    fingerprints and certificate scores, never gold answers.
    """

    entries: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def add(self, operator_id: str, certificate: EpistemicCertificate) -> None:
        self.entries.setdefault(str(operator_id), []).append(certificate.to_dict())

    def transfer_support(self, operator_id: str, fingerprint: TaskFingerprint) -> float:
        rows = self.entries.get(str(operator_id), [])
        if not rows:
            return 0.0
        families = {
            str(row.get("fingerprint", {}).get("interface_family", ""))
            for row in rows
            if row.get("accepted") or row.get("status") in {"accepted", "provisional"}
        }
        families.discard("")
        if not families:
            return 0.0
        support = math.log1p(len(families)) / math.log(5)
        if fingerprint.interface_family in families:
            support += 0.15
        return float(min(1.0, support))


def fingerprint_task(
    *,
    signature_hint: str,
    interface_description: str,
    observations: Sequence[Any],
) -> TaskFingerprint:
    """Build a benchmark-free signature from observable I/O only."""

    input_kinds: list[str] = []
    target_kinds: list[str] = []
    context_keys: set[str] = set()
    input_keys: set[str] = set()
    numeric_hits = 0
    sequence_hits = 0
    table_hits = 0
    text_hits = 0
    total_cells = 0
    payload_rows: list[dict[str, Any]] = []

    for obs in observations:
        inputs = getattr(obs, "inputs", None)
        target = getattr(obs, "target", None)
        context = getattr(obs, "context", {}) or {}
        input_kinds.append(type(inputs).__name__)
        target_kinds.append(type(target).__name__)
        context_keys.update(str(k) for k in getattr(context, "keys", lambda: [])())
        if isinstance(inputs, Mapping):
            input_keys.update(str(k) for k in inputs.keys())
        elif _looks_like_dataframe(inputs):
            table_hits += 1
        for value in _flatten_observable(inputs) + _flatten_observable(target):
            total_cells += 1
            if _is_number(value):
                numeric_hits += 1
            elif isinstance(value, str):
                if _looks_sequence_like(value):
                    sequence_hits += 1
                if len(value.strip()) >= 16:
                    text_hits += 1
        payload_rows.append(
            {
                "input_kind": type(inputs).__name__,
                "target_kind": type(target).__name__,
                "context_keys": sorted(str(k) for k in getattr(context, "keys", lambda: [])()),
                "input_keys": sorted(str(k) for k in inputs.keys()) if isinstance(inputs, Mapping) else [],
            }
        )

    total = max(1, total_cells)
    family = _infer_interface_family(signature_hint, interface_description, input_kinds, target_kinds)
    observable_payload = {
        "signature_hint": _scrub_benchmark_tokens(signature_hint),
        "family": family,
        "input_kinds": sorted(set(input_kinds)),
        "target_kinds": sorted(set(target_kinds)),
        "context_keys": sorted(context_keys),
        "input_keys": sorted(input_keys),
        "shape_rows": payload_rows[:32],
    }
    raw = json.dumps(observable_payload, sort_keys=True, ensure_ascii=True, default=str)
    return TaskFingerprint(
        signature_hint=signature_hint,
        interface_family=family,
        input_kind=_mode(input_kinds),
        target_kind=_mode(target_kinds),
        context_keys=tuple(sorted(context_keys)[:48]),
        input_keys=tuple(sorted(input_keys)[:64]),
        n_observations=len(observations),
        numeric_density=round(numeric_hits / total, 6),
        sequence_density=round(sequence_hits / total, 6),
        table_density=round(table_hits / max(1, len(observations)), 6),
        text_density=round(text_hits / total, 6),
        observable_hash=hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16],
    )


def certify_epistemic_result(
    *,
    adapter_name: str,
    signature_hint: str,
    interface_description: str,
    observations: Sequence[Any],
    winners: Sequence[tuple[Any, Any]],
    programs: Sequence[Any],
    proposals_raw: Sequence[Mapping[str, Any]] | None = None,
    ledger: EpistemicLedger | None = None,
) -> EpistemicCertificate:
    """Score whether a run looks like universal induction, not local patching."""

    fingerprint = fingerprint_task(
        signature_hint=signature_hint,
        interface_description=interface_description,
        observations=observations,
    )
    if not winners:
        return EpistemicCertificate(
            fingerprint=fingerprint,
            status="rejected",
            universal_score=-1.0,
            compression_gain=0.0,
            stability_score=0.0,
            transfer_support=0.0,
            complexity_penalty=0.0,
            leakage_penalty=0.0,
            best_loss=1.0,
            reference_loss=1.0,
            evidence_count=len(observations),
            winner_family="none",
            rejected_reasons=("no winning hypothesis program",),
        )

    best_program, best_score = winners[0]
    best_loss = _safe_float(getattr(best_score, "loss_mean", 1.0), 1.0)
    scored_losses = [
        _safe_float(getattr(score, "loss_mean", 1.0), 1.0)
        for _program, score in winners
    ]
    if not scored_losses:
        scored_losses = [best_loss]
    reference_loss = max(0.05, _median(scored_losses + [1.0]))
    compression_gain = max(0.0, math.log((reference_loss + _EPS) / (best_loss + _EPS)))
    complexity = _safe_float(getattr(best_program, "complexity", 2.0), 2.0)
    code = str(getattr(best_program, "code", "") or "")
    description = str(getattr(best_program, "description", "") or "")
    complexity_penalty = 0.08 * complexity + 0.015 * math.log1p(len(code))
    leakage_penalty = benchmark_leakage_penalty(
        " ".join([str(getattr(best_program, "name", "")), description, code]),
        adapter_name=adapter_name,
    )
    stability_score = _stability_from_scores(winners)
    winner_family = infer_operator_family(best_program)
    transfer_support = _transfer_support(
        best_program,
        fingerprint,
        ledger=ledger,
        proposals_raw=proposals_raw or (),
    )
    universal_score = (
        compression_gain
        + 0.40 * stability_score
        + 0.35 * transfer_support
        - complexity_penalty
        - leakage_penalty
    )

    accepted: list[str] = []
    rejected: list[str] = []
    if compression_gain >= 0.20:
        accepted.append("held-out loss is compressed relative to candidate reference")
    else:
        rejected.append("compression gain is too small")
    if leakage_penalty <= 0.20:
        accepted.append("no strong benchmark-name leakage in winning program")
    else:
        rejected.append("winner text/code contains benchmark-specific tokens")
    if best_loss <= 0.25:
        accepted.append("best held-out loss is low enough for reusable evidence")
    else:
        rejected.append("best held-out loss remains high")
    if fingerprint.n_observations >= 3:
        accepted.append("certificate has multiple observations")
    else:
        rejected.append("too few observations for a stable certificate")

    if rejected:
        status = "provisional" if universal_score > 0.0 and leakage_penalty <= 0.25 else "rejected"
    else:
        status = "accepted"
    return EpistemicCertificate(
        fingerprint=fingerprint,
        status=status,
        universal_score=float(universal_score),
        compression_gain=float(compression_gain),
        stability_score=float(stability_score),
        transfer_support=float(transfer_support),
        complexity_penalty=float(complexity_penalty),
        leakage_penalty=float(leakage_penalty),
        best_loss=float(best_loss),
        reference_loss=float(reference_loss),
        evidence_count=len(observations),
        winner_family=winner_family,
        accepted_reasons=tuple(accepted),
        rejected_reasons=tuple(rejected),
        diagnostics={
            "adapter_name": adapter_name,
            "winner_name": str(getattr(best_program, "name", "")),
            "n_programs": len(programs),
            "n_proposals": len(proposals_raw or ()),
        },
    )


def benchmark_leakage_penalty(text: str, *, adapter_name: str = "") -> float:
    """Penalize solutions that mention benchmark identity instead of evidence."""

    hay = str(text or "").lower()
    adapter_tokens = {
        token
        for token in re.split(r"[^a-z0-9]+", str(adapter_name).lower())
        if len(token) >= 3
    }
    tokens = _BENCHMARK_TOKENS | adapter_tokens
    hits = sorted(token for token in tokens if token and token in hay)
    if not hits:
        return 0.0
    # A single adapter-like token in metadata is suspicious but not fatal; many
    # explicit benchmark words are strong evidence of local patching.
    return float(min(1.0, 0.18 + 0.18 * len(set(hits))))


def infer_operator_family(program: Any) -> str:
    tags = tuple(str(t) for t in getattr(program, "tags", ()) or ())
    if tags:
        for tag in tags:
            if tag not in {"proposed", "candidate"}:
                return tag
    name = str(getattr(program, "name", "") or "").lower()
    for prefix in (
        "dpsr",
        "darwin",
        "nova",
        "syndrome",
        "residual",
        "typed",
        "self_layer",
        "self_module",
        "seed",
        "composition",
        "branch",
    ):
        if prefix in name:
            return prefix
    return "program"


def _transfer_support(
    program: Any,
    fingerprint: TaskFingerprint,
    *,
    ledger: EpistemicLedger | None,
    proposals_raw: Sequence[Mapping[str, Any]],
) -> float:
    tags = {str(t) for t in getattr(program, "tags", ()) or ()}
    support = 0.0
    if "self_module_trusted" in tags or "self_layer_trusted" in tags:
        support = max(support, 0.7)
    elif any("trusted" in tag for tag in tags):
        support = max(support, 0.55)
    if any(str(p.get("kind", "")) in {"darwin", "darwin_trace", "ip_genome_context"} for p in proposals_raw):
        support = max(support, 0.25)
    if ledger is not None:
        support = max(support, ledger.transfer_support(str(getattr(program, "name", "")), fingerprint))
    return float(min(1.0, support))


def _infer_interface_family(
    signature_hint: str,
    interface_description: str,
    input_kinds: Sequence[str],
    target_kinds: Sequence[str],
) -> str:
    signature = str(signature_hint).lower()
    desc = _scrub_benchmark_tokens(interface_description).lower()
    if "dataframe" in signature or "df" in signature or "table" in desc:
        return "table_analysis"
    if "current: str" in signature or "string" in desc or "str" in set(input_kinds):
        return "state_transition"
    if "inputs: dict" in signature or "scalar" in desc or "float" in set(target_kinds):
        return "numeric_law"
    if "parent" in signature or "predict" in signature:
        return "structured_state"
    return "generic_evidence"


def _scrub_benchmark_tokens(text: str) -> str:
    out = str(text or "")
    for token in sorted(_BENCHMARK_TOKENS, key=len, reverse=True):
        out = re.sub(re.escape(token), "<bench>", out, flags=re.IGNORECASE)
    return out


def _flatten_observable(value: Any, limit: int = 256) -> list[Any]:
    out: list[Any] = []

    def visit(v: Any) -> None:
        if len(out) >= limit:
            return
        if isinstance(v, Mapping):
            for item in v.values():
                visit(item)
            return
        if isinstance(v, (list, tuple, set)):
            for item in list(v)[:limit]:
                visit(item)
            return
        if _looks_like_dataframe(v):
            out.append("dataframe")
            return
        out.append(v)

    visit(value)
    return out


def _looks_like_dataframe(value: Any) -> bool:
    return hasattr(value, "columns") and hasattr(value, "shape")


def _looks_sequence_like(value: str) -> bool:
    s = value.strip()
    return bool(s) and len(s) <= 64 and bool(re.fullmatch(r"[A-Za-z0-9_ -]+", s))


def _is_number(value: Any) -> bool:
    try:
        if isinstance(value, bool):
            return False
        x = float(value)
        return math.isfinite(x)
    except Exception:
        return False


def _mode(values: Sequence[str]) -> str:
    if not values:
        return "unknown"
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _median(values: Iterable[float]) -> float:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return 1.0
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def _safe_float(value: Any, default: float) -> float:
    try:
        out = float(value)
        return out if math.isfinite(out) else default
    except Exception:
        return default


def _stability_from_scores(winners: Sequence[tuple[Any, Any]]) -> float:
    if len(winners) < 2:
        loss = _safe_float(getattr(winners[0][1], "loss_mean", 1.0), 1.0) if winners else 1.0
        return float(max(0.0, 1.0 - loss))
    losses = [_safe_float(getattr(score, "loss_mean", 1.0), 1.0) for _program, score in winners[:5]]
    spread = max(losses) - min(losses)
    best = min(losses)
    return float(max(0.0, 1.0 - best - 0.25 * spread))
