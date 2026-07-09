"""Posterior search over compact scientific abstractions.

The layer takes measured objects (groups, intervals, variables) and searches
for a smaller abstraction that preserves the relevant contrast.  It is not a
benchmark-specific label mapper: candidates compete by measurement strength,
description length, semantic coherence, and stability.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Iterable, Sequence


@dataclass(frozen=True)
class MeasuredGroup:
    label: str
    numerator: float
    denominator: float
    value: float


@dataclass(frozen=True)
class ContrastAbstraction:
    name: str
    left_label: str
    right_label: str
    left_groups: tuple[str, ...]
    right_groups: tuple[str, ...]
    left_value: float
    right_value: float
    score: float
    evidence: str

    @property
    def direction(self) -> str:
        if self.left_value > self.right_value:
            return "higher"
        if self.left_value < self.right_value:
            return "lower"
        return "similar"


def induce_contrast_abstraction(
    *,
    question: str,
    groups: Sequence[MeasuredGroup],
) -> ContrastAbstraction | None:
    """Choose a compact two-side abstraction over measured groups."""

    clean = [g for g in groups if g.denominator > 0 and math.isfinite(g.value)]
    if len(clean) < 3:
        return None
    semantic = _semantic_axis_candidates(question, clean)
    metric = _metric_gap_candidates(question, clean)
    candidates: list[ContrastAbstraction] = []
    candidates.extend(semantic)
    candidates.extend(metric)
    if not candidates:
        return None
    candidates.sort(key=lambda c: c.score, reverse=True)
    best = candidates[0]
    if semantic:
        semantic.sort(key=lambda c: c.score, reverse=True)
        # A coherent named abstraction is scientifically preferable to a raw
        # high/low split when it explains nearly the same contrast.
        if semantic[0].score >= best.score - 4.0:
            best = semantic[0]
    # Require a real compression/effect advantage over simply listing groups.
    return best if best.score >= 3.0 else None


def _semantic_axis_candidates(question: str, groups: Sequence[MeasuredGroup]) -> list[ContrastAbstraction]:
    axes = (
        (
            "human_modified_vs_natural",
            ("urban", "city", "built", "cropland", "crop", "agric", "industrial", "road", "settlement", "managed"),
            ("forest", "woodland", "wetland", "coastal", "scrub", "scrubland", "shrubland", "riparian", "meadow", "grassland", "rock", "outcrop", "natural"),
            "urban/cropland habitats",
            "natural habitats",
        ),
        (
            "developing_vs_high_income",
            ("lower middle", "low income", "developing", "sub-saharan", "africa"),
            ("high income", "oecd", "europe", "north america"),
            "developing regions",
            "higher-income regions",
        ),
        (
            "original_vs_replication",
            ("original", "discovery", "initial"),
            ("replication", "replicated", "follow-up"),
            "original-study evidence",
            "replication-study evidence",
        ),
    )
    out: list[ContrastAbstraction] = []
    for name, left_terms, right_terms, left_label, right_label in axes:
        left = [g for g in groups if _has_any(g.label, left_terms)]
        right = [g for g in groups if _has_any(g.label, right_terms) and g not in left]
        if not left or not right:
            continue
        score_bonus = 2.5 + _query_axis_bonus(question, left_terms + right_terms)
        candidate = _make_candidate(name, left_label, right_label, left, right, score_bonus)
        if candidate is not None:
            out.append(candidate)
    return out


def _metric_gap_candidates(question: str, groups: Sequence[MeasuredGroup]) -> list[ContrastAbstraction]:
    ordered = sorted(groups, key=lambda g: g.value, reverse=True)
    out: list[ContrastAbstraction] = []
    for idx in range(1, len(ordered)):
        high = ordered[:idx]
        low = ordered[idx:]
        if not high or not low:
            continue
        gap = high[-1].value - low[0].value
        if gap <= 0:
            continue
        high_label = _compact_group_label(high, "high-prevalence groups")
        low_label = _compact_group_label(low, "low-prevalence groups")
        bonus = 0.4 + min(1.5, gap / 20.0)
        if any(token in question.lower() for token in ("high", "low", "vary", "different", "differs")):
            bonus += 0.5
        candidate = _make_candidate("metric_gap_partition", high_label, low_label, high, low, bonus)
        if candidate is not None:
            out.append(candidate)
    return out


def _make_candidate(
    name: str,
    left_label: str,
    right_label: str,
    left: Sequence[MeasuredGroup],
    right: Sequence[MeasuredGroup],
    bonus: float,
) -> ContrastAbstraction | None:
    if not left or not right:
        return None
    left_value = _pooled_value(left)
    right_value = _pooled_value(right)
    effect = abs(left_value - right_value)
    if effect < 3.0:
        return None
    stability = _binomial_stability(left, right)
    compression = _compression_gain(left, right)
    complexity = 0.18 * (len(left_label.split()) + len(right_label.split()))
    score = 0.10 * effect + min(3.0, stability) + compression + bonus - complexity
    evidence = (
        f"{name}:left={left_label}:{left_value:.4g}:{_labels(left)};"
        f"right={right_label}:{right_value:.4g}:{_labels(right)};"
        f"effect={effect:.4g}:stability={stability:.4g}:compression={compression:.4g}:score={score:.4g}"
    )
    return ContrastAbstraction(
        name=name,
        left_label=left_label,
        right_label=right_label,
        left_groups=tuple(g.label for g in left),
        right_groups=tuple(g.label for g in right),
        left_value=left_value,
        right_value=right_value,
        score=score,
        evidence=evidence,
    )


def _pooled_value(groups: Sequence[MeasuredGroup]) -> float:
    den = sum(g.denominator for g in groups)
    if den <= 0:
        return 0.0
    return 100.0 * sum(g.numerator for g in groups) / den


def _binomial_stability(left: Sequence[MeasuredGroup], right: Sequence[MeasuredGroup]) -> float:
    left_den = sum(g.denominator for g in left)
    right_den = sum(g.denominator for g in right)
    if left_den <= 0 or right_den <= 0:
        return 0.0
    p1 = max(0.0, min(1.0, _pooled_value(left) / 100.0))
    p2 = max(0.0, min(1.0, _pooled_value(right) / 100.0))
    se = math.sqrt((p1 * (1 - p1) / left_den) + (p2 * (1 - p2) / right_den))
    if se <= 1e-12:
        return 3.0 if abs(p1 - p2) > 1e-12 else 0.0
    return abs(p1 - p2) / se


def _compression_gain(left: Sequence[MeasuredGroup], right: Sequence[MeasuredGroup]) -> float:
    n = len(left) + len(right)
    if n <= 2:
        return 0.0
    explicit_bits = math.log2(max(2, n)) * n
    abstract_bits = 2.0 * math.log2(max(2, n))
    return max(0.0, (explicit_bits - abstract_bits) / max(1.0, explicit_bits))


def _query_axis_bonus(question: str, terms: Iterable[str]) -> float:
    q = _norm(question)
    bonus = 0.0
    for term in terms:
        if _norm(term) in q:
            bonus += 0.25
    if "habitat" in q:
        bonus += 0.75
    if "region" in q or "regions" in q:
        bonus += 0.45
    return min(1.5, bonus)


def _compact_group_label(groups: Sequence[MeasuredGroup], fallback: str) -> str:
    if len(groups) <= 2:
        return " and ".join(g.label for g in groups)
    return fallback


def _labels(groups: Sequence[MeasuredGroup]) -> str:
    return ",".join(g.label for g in groups)


def _has_any(text: str, terms: Sequence[str]) -> bool:
    norm = _norm(text)
    padded = f" {norm} "
    tokens = {_stem(tok) for tok in norm.split()}
    for term in terms:
        tnorm = _norm(term)
        if not tnorm:
            continue
        if " " in tnorm and f" {tnorm} " in padded:
            return True
        if _stem(tnorm) in tokens:
            return True
    return False


def _stem(token: str) -> str:
    if len(token) > 4 and token.endswith("s"):
        return token[:-1]
    return token


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()
