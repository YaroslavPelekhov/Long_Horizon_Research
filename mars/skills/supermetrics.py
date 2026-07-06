"""Universal internal metrics for self-building hypothesis systems.

These metrics are intentionally benchmark-agnostic.  Official benchmark scores
remain the external target; supermetrics measure whether the system is doing the
scientific work that should transfer across benchmarks: execute, refute, localize
residuals, concentrate posterior mass, ground the report, and reuse learned
structure.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class SuperMetricReport:
    benchmark: str
    n_observations: int
    n_proposed: int
    n_valid: int
    execution_validity: float
    heldout_fit: float
    exactness: float
    mdl_efficiency: float
    posterior_concentration: float
    residual_localization: float
    transfer_reuse: float
    report_grounding: float
    universal_score: float
    best_name: str = ""
    best_loss: float | None = None
    best_exact_rate: float | None = None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["notes"] = list(self.notes)
        return out


DEFAULT_WEIGHTS: Mapping[str, float] = {
    "execution_validity": 0.14,
    "heldout_fit": 0.22,
    "exactness": 0.14,
    "mdl_efficiency": 0.10,
    "posterior_concentration": 0.10,
    "residual_localization": 0.10,
    "transfer_reuse": 0.08,
    "report_grounding": 0.12,
}


def compute_cpi_supermetrics(
    *,
    benchmark: str,
    n_observations: int,
    n_proposed: int,
    n_valid: int,
    winners: Sequence[tuple[Any, Any]],
    report: str,
    errors: Sequence[str] = (),
    wall_time_s: float | None = None,
    rounds: int = 1,
    weights: Mapping[str, float] = DEFAULT_WEIGHTS,
) -> SuperMetricReport:
    """Compute a universal diagnostic score for any CPI-like result."""

    best_program, best_score = winners[0] if winners else (None, None)
    best_loss = _float_attr(best_score, "loss_mean")
    best_exact = _float_attr(best_score, "exact_rate")
    best_mdl = _float_attr(best_score, "mdl_score")
    valid_den = max(1, n_proposed, n_valid)

    execution_validity = _clamp(n_valid / valid_den)
    heldout_fit = 0.0 if best_loss is None else _clamp(1.0 - best_loss)
    exactness = 0.0 if best_exact is None else _clamp(best_exact)
    mdl_efficiency = _mdl_efficiency(best_mdl, best_loss, _float_attr(best_score, "complexity"))
    posterior_concentration = _posterior_concentration([score for _program, score in winners])
    residual_localization = _residual_localization(errors, best_loss)
    transfer_reuse = _transfer_reuse(winners, errors)
    report_grounding = _report_grounding(report)

    metric_values = {
        "execution_validity": execution_validity,
        "heldout_fit": heldout_fit,
        "exactness": exactness,
        "mdl_efficiency": mdl_efficiency,
        "posterior_concentration": posterior_concentration,
        "residual_localization": residual_localization,
        "transfer_reuse": transfer_reuse,
        "report_grounding": report_grounding,
    }
    universal_score = _weighted_mean(metric_values, weights)
    notes = _notes(
        n_observations=n_observations,
        n_proposed=n_proposed,
        n_valid=n_valid,
        winners=winners,
        errors=errors,
        wall_time_s=wall_time_s,
        rounds=rounds,
    )
    return SuperMetricReport(
        benchmark=benchmark,
        n_observations=int(n_observations),
        n_proposed=int(n_proposed),
        n_valid=int(n_valid),
        execution_validity=round(execution_validity, 4),
        heldout_fit=round(heldout_fit, 4),
        exactness=round(exactness, 4),
        mdl_efficiency=round(mdl_efficiency, 4),
        posterior_concentration=round(posterior_concentration, 4),
        residual_localization=round(residual_localization, 4),
        transfer_reuse=round(transfer_reuse, 4),
        report_grounding=round(report_grounding, 4),
        universal_score=round(universal_score, 4),
        best_name=str(getattr(best_program, "name", "")) if best_program is not None else "",
        best_loss=None if best_loss is None else round(best_loss, 6),
        best_exact_rate=None if best_exact is None else round(best_exact, 6),
        notes=notes,
    )


def aggregate_supermetrics(reports: Iterable[SuperMetricReport | Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate supermetrics across benchmarks without knowing benchmark names."""

    rows = [r.to_dict() if isinstance(r, SuperMetricReport) else dict(r) for r in reports]
    if not rows:
        return {"n": 0, "mean_universal_score": 0.0, "by_metric": {}, "epistemic": {}}
    metric_names = [
        "execution_validity",
        "heldout_fit",
        "exactness",
        "mdl_efficiency",
        "posterior_concentration",
        "residual_localization",
        "transfer_reuse",
        "report_grounding",
        "universal_score",
    ]
    by_metric = {}
    for name in metric_names:
        vals = [float(row.get(name, 0.0)) for row in rows if _is_number(row.get(name, None))]
        by_metric[name] = round(sum(vals) / len(vals), 4) if vals else 0.0
    return {
        "n": len(rows),
        "mean_universal_score": by_metric["universal_score"],
        "by_metric": by_metric,
        "epistemic": _aggregate_epistemic_certificates(rows),
        "benchmarks": {str(row.get("benchmark", f"row_{i}")): row for i, row in enumerate(rows)},
    }


def official_gap_metric(
    *,
    weak_score: float,
    strong_score: float,
    system_score: float,
    higher_is_better: bool = True,
) -> dict[str, float | bool]:
    """Normalize progress against weak and strong official benchmark scores."""

    if not higher_is_better:
        weak_score, strong_score, system_score = -weak_score, -strong_score, -system_score
    denom = abs(strong_score - weak_score)
    if denom <= 1e-12:
        progress = 1.0 if system_score >= strong_score else 0.0
    else:
        progress = (system_score - weak_score) / denom
    return {
        "weak_score": round(float(weak_score), 6),
        "strong_score": round(float(strong_score), 6),
        "system_score": round(float(system_score), 6),
        "progress_weak_to_strong": round(progress, 4),
        "beats_weak": bool(system_score >= weak_score),
        "beats_strong": bool(system_score >= strong_score),
    }


def _float_attr(obj: Any, name: str) -> float | None:
    if obj is None:
        return None
    value = getattr(obj, name, None)
    if value is None and isinstance(obj, Mapping):
        value = obj.get(name)
    if not _is_number(value):
        return None
    return float(value)


def _is_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except Exception:
        return False


def _aggregate_epistemic_certificates(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    certificates: list[Mapping[str, Any]] = []
    for row in rows:
        cert = row.get("epistemic_certificate")
        if isinstance(cert, Mapping):
            certificates.append(cert)
    if not certificates:
        return {
            "n": 0,
            "accepted": 0,
            "provisional": 0,
            "rejected": 0,
            "acceptance_rate": 0.0,
            "mean_epistemic_score": 0.0,
            "mean_compression_gain": 0.0,
            "mean_leakage_penalty": 0.0,
            "mean_transfer_support": 0.0,
            "interface_families": {},
        }

    statuses: dict[str, int] = {"accepted": 0, "provisional": 0, "rejected": 0}
    families: dict[str, int] = {}
    for cert in certificates:
        status = str(cert.get("status", "rejected"))
        statuses[status] = statuses.get(status, 0) + 1
        family = str((cert.get("fingerprint") or {}).get("interface_family", "unknown"))
        families[family] = families.get(family, 0) + 1

    return {
        "n": len(certificates),
        "accepted": statuses.get("accepted", 0),
        "provisional": statuses.get("provisional", 0),
        "rejected": statuses.get("rejected", 0),
        "acceptance_rate": round(statuses.get("accepted", 0) / max(1, len(certificates)), 4),
        "mean_epistemic_score": _mean_cert_field(certificates, "universal_score"),
        "mean_compression_gain": _mean_cert_field(certificates, "compression_gain"),
        "mean_leakage_penalty": _mean_cert_field(certificates, "leakage_penalty"),
        "mean_transfer_support": _mean_cert_field(certificates, "transfer_support"),
        "interface_families": dict(sorted(families.items())),
    }


def _mean_cert_field(certificates: Sequence[Mapping[str, Any]], field: str) -> float:
    values = [float(cert.get(field, 0.0)) for cert in certificates if _is_number(cert.get(field))]
    if not values:
        return 0.0
    return round(sum(values) / len(values), 4)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if not math.isfinite(value):
        return lo
    return min(hi, max(lo, float(value)))


def _mdl_efficiency(mdl: float | None, loss: float | None, complexity: float | None) -> float:
    if mdl is None:
        if loss is None:
            return 0.0
        mdl = loss + 0.05 * (complexity if complexity is not None else 2.0)
    return _clamp(1.0 / (1.0 + max(0.0, mdl)))


def _posterior_concentration(scores: Sequence[Any]) -> float:
    if not scores:
        return 0.0
    costs = []
    for score in scores:
        mdl = _float_attr(score, "mdl_score")
        loss = _float_attr(score, "loss_mean")
        complexity = _float_attr(score, "complexity")
        if mdl is None:
            mdl = (loss if loss is not None else 1.0) + 0.05 * (complexity if complexity is not None else 2.0)
        costs.append(float(mdl))
    if len(costs) == 1:
        return 1.0
    best = min(costs)
    weights = [math.exp(-(c - best)) for c in costs]
    total = sum(weights)
    if total <= 0:
        return 0.0
    probs = [w / total for w in weights]
    entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs)
    entropy_conc = 1.0 - entropy / math.log(len(probs))
    best_prob = max(probs)
    return _clamp(0.5 * entropy_conc + 0.5 * best_prob)


def _residual_localization(errors: Sequence[str], best_loss: float | None) -> float:
    if best_loss is not None and best_loss <= 1e-9:
        return 1.0
    if not errors:
        return 0.25 if best_loss is not None else 0.0
    typed = 0
    residual = 0
    for item in errors:
        low = str(item).lower()
        if ":" in low:
            typed += 1
        if any(tok in low for tok in ("residual", "counterexample", "schema", "type", "shape", "loss", "failed")):
            residual += 1
    return _clamp(0.35 + 0.35 * typed / len(errors) + 0.30 * residual / len(errors))


def _transfer_reuse(winners: Sequence[tuple[Any, Any]], errors: Sequence[str]) -> float:
    signals = 0
    total = 0
    for program, _score in winners:
        total += 1
        tags = tuple(str(t).lower() for t in getattr(program, "tags", ()) or ())
        text = " ".join([str(getattr(program, "name", "")), str(getattr(program, "description", "")), *tags]).lower()
        if any(tok in text for tok in ("trusted", "self_layer", "self_module", "operator", "residual", "typed_prior", "dpsr")):
            signals += 1
    for item in errors:
        low = str(item).lower()
        if any(tok in low for tok in ("trusted library", "promoted self", "self_layer", "self module", "operator")):
            signals += 1
            total += 1
    if total == 0:
        return 0.0
    return _clamp(signals / total)


def _report_grounding(report: str) -> float:
    if not report or not str(report).strip():
        return 0.0
    low = str(report).lower()
    score = 0.2
    if any(tok in low for tok in ("evidence:", "evidence=", "slot_contract", "residual", "held-out", "workflow")):
        score += 0.35
    if any(tok in low for tok in ("variable=", "time=", "loss=", "corr(", "value=", "exact", "posterior=")):
        score += 0.25
    if re.search(r"\b\d+(?:\.\d+)?\b", low):
        score += 0.10
    if len(str(report).strip()) >= 40:
        score += 0.10
    return _clamp(score)


def _weighted_mean(values: Mapping[str, float], weights: Mapping[str, float]) -> float:
    total_w = sum(float(weights.get(k, 0.0)) for k in values)
    if total_w <= 0:
        return 0.0
    return _clamp(sum(values[k] * float(weights.get(k, 0.0)) for k in values) / total_w)


def _notes(
    *,
    n_observations: int,
    n_proposed: int,
    n_valid: int,
    winners: Sequence[tuple[Any, Any]],
    errors: Sequence[str],
    wall_time_s: float | None,
    rounds: int,
) -> tuple[str, ...]:
    notes: list[str] = []
    if n_observations <= 1:
        notes.append("low_observation_count")
    if n_proposed > 0 and n_valid == 0:
        notes.append("no_valid_programs")
    if winners:
        best_loss = _float_attr(winners[0][1], "loss_mean")
        if best_loss is not None and best_loss > 0.25:
            notes.append("high_best_residual")
    if errors:
        notes.append(f"errors={len(errors)}")
    if wall_time_s is not None and wall_time_s > 120:
        notes.append("slow_run")
    if rounds > 1:
        notes.append(f"repair_rounds={rounds}")
    return tuple(notes[:8])
