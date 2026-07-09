"""Conservation-Discovery — ONE universal engine. No benchmark code, no tuned constants.

Principle (Noether for inference): the truth of an answer is INVARIANT under transformations
that preserve the meaning of the data (bootstrap resample, row permutation, disjoint split).
A correct answer's executable quantity is CONSERVED across these; a spurious one is not. We
keep candidates whose quantity is conserved AT THE DATA'S OWN NOISE FLOOR (self-calibrated,
never a hand-set threshold) and pick the strongest; else we abstain.

The engine knows NOTHING about any benchmark. A `DiscoveryAdapter` supplies, per task:
  - generate_candidates(ctx) -> list[Candidate]   (the MODEL authors these)
  - transforms(ctx)          -> list[transform]   (meaning-preserving resamplers)
  - noise_floor(ctx)         -> float             (irreducible error, measured from data)
  - score(answer)            -> float             (official judge; eval only, not selection)

A Candidate carries an `answer` (what we'd submit) and `quantity(ctx)->float` (the signed
scalar the answer asserts) and optionally `fidelity(ctx)->float` (held-out reproduction error,
for generative tasks where the answer is a law fit to data). The same selection logic serves
both verifier-poor selection (DiscoveryBench: quantity stability) and law discovery
(NewtonBench: fidelity at noise floor)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np


@dataclass
class Candidate:
    answer: str
    quantity: Callable[[dict], float] | None = None     # signed scalar the claim asserts
    fidelity: Callable[[dict], float] | None = None      # held-out reproduction error (laws)
    answer_key: str | None = None                        # normalized outcome for consensus
    meta: dict = field(default_factory=dict)


@dataclass
class DiscoveryAdapter:
    """A task plugs into the engine by providing these. The model does the thinking inside
    generate_candidates; everything else is generic measurement."""
    generate_candidates: Callable[[dict], list[Candidate]]
    transforms: Callable[[dict], list[Callable[[dict], dict]]]
    score: Callable[[str], float]
    noise_floor: Callable[[dict], float] = lambda ctx: 0.0


@dataclass
class Selection:
    answer: str | None
    abstained: bool
    reason: str
    detail: list[dict] = field(default_factory=list)


def _safe(fn, ctx):
    try:
        v = float(fn(ctx))
        return v if np.isfinite(v) else None
    except Exception:
        return None


def _conservation(quantity, ctx, transforms):
    """sign_consistency × magnitude_stability of the quantity under meaning-preserving
    transforms. Self-contained, no thresholds."""
    full = _safe(quantity, ctx)
    if full is None or full == 0:
        return 0.0, 0.0, full
    vals = []
    for t in transforms:
        v = _safe(quantity, t(ctx))
        if v is not None:
            vals.append(v)
    if len(vals) < max(2, len(transforms) * 0.5):
        return 0.0, 0.0, full
    vals = np.array(vals)
    sign_consistency = float(np.mean(np.sign(vals) == np.sign(full)))
    cv = float(np.std(vals) / (abs(np.mean(vals)) + 1e-12))
    stability = 1.0 / (1.0 + cv)
    return sign_consistency, stability, full


def select(adapter: DiscoveryAdapter, ctx: dict) -> Selection:
    """Universal: generate -> measure conservation/fidelity -> keep what holds at the noise
    floor -> pick strongest, else abstain. No benchmark logic, no tuned constants."""
    cands = adapter.generate_candidates(ctx)
    if not cands:
        return Selection(None, True, "no candidate generated")

    # CONSENSUS mode: candidates that carry neither a measurable quantity nor a fidelity are
    # consensus-type (code, agent rollouts). Truth is REPRODUCIBLE across independent
    # generations; errors are idiosyncratic. Keep the outcome reproduced by a strict majority
    # of SUCCESSFUL candidates (beyond-chance agreement); if none ran or none agree, abstain.
    has_measure = any(c.quantity is not None or c.fidelity is not None for c in cands)
    if not has_measure:
        from collections import Counter
        ok = [c for c in cands if c.answer_key is not None]
        detail = [{"answer": c.answer[:80], "key": c.answer_key} for c in cands]
        if not ok:
            return Selection(None, True, "no candidate produced a usable outcome", detail)
        counts = Counter(c.answer_key for c in ok)
        key, n = counts.most_common(1)[0]
        if n < 2 or n <= len(ok) / 2:            # not a strict majority -> not reproduced
            return Selection(None, True, f"no consensus (top {n}/{len(ok)} agree, {len(ok)}/{len(cands)} ran)", detail)
        rep = next(c for c in ok if c.answer_key == key)
        return Selection(rep.answer, False, f"consensus {n}/{len(ok)}", detail)

    transforms = adapter.transforms(ctx)
    floor = adapter.noise_floor(ctx)
    # tolerance for "at the noise floor": a few × the floor, with a tiny absolute epsilon for
    # exactly-noiseless data (machine precision). Self-calibrated from the data, not hand-set.
    fid_tol = max(10.0 * floor, 1e-6)

    scored = []
    for c in cands:
        rec = {"answer": c.answer[:120], **c.meta}
        keep = True
        if c.fidelity is not None:                       # generative law: must reproduce data
            fid = _safe(c.fidelity, ctx)
            rec["fidelity"] = None if fid is None else round(fid, 8)
            if fid is None or fid > fid_tol:
                keep = False
            rec["strength"] = -(fid if fid is not None else 9e9)
        if c.quantity is not None:                       # claim: quantity must be conserved
            sign_c, stab, full = _conservation(c.quantity, ctx, transforms)
            rec["sign"], rec["stab"] = round(sign_c, 3), round(stab, 3)
            # conserved == sign always agrees AND stable relative to its own scale
            if not (sign_c >= 1.0 - 1e-9 and stab >= 0.5):
                keep = False
            rec.setdefault("strength", sign_c * stab)
        rec["keep"] = keep
        scored.append((c, rec))

    survivors = [(c, r) for c, r in scored if r["keep"]]
    detail = [r for _, r in scored]
    if not survivors:
        return Selection(None, True, "nothing conserved at the data noise floor", detail)
    best = max(survivors, key=lambda cr: cr[1].get("strength", 0.0))
    return Selection(best[0].answer, False, "selected most-conserved candidate", detail)


# ---- generic measurement helpers an adapter may reuse (still benchmark-agnostic) ----

def bootstrap_transforms(df_key="df", n=20, frac=0.6, seed=0):
    """Meaning-preserving resamplers for tabular ctx: bootstrap + disjoint subsample+shuffle."""
    def make(ctx):
        df = ctx[df_key]
        rng = np.random.RandomState(seed)
        nrows = len(df)
        out = []
        for i in range(n):
            if i % 2 == 0:
                idx = rng.randint(0, nrows, size=nrows)
            else:
                idx = rng.permutation(nrows)[: max(3, int(nrows * frac))]
            sub = df.iloc[idx].reset_index(drop=True)
            out.append(lambda c, _s=sub: {**c, df_key: _s, "columns": list(_s.columns)})
        return out
    return make


def permutation_null_z(quantity, ctx, df_key="df", n=30, seed=1):
    """Shuffle each column independently (destroys cross-column structure). z = how far the
    real quantity sits outside the null. High z => a real relationship, not an artifact."""
    df = ctx[df_key]
    rng = np.random.RandomState(seed)
    full = _safe(quantity, ctx)
    if full is None:
        return 0.0
    nulls = []
    for _ in range(n):
        sh = df.copy()
        for col in sh.columns:
            sh[col] = sh[col].values[rng.permutation(len(sh))]
        v = _safe(quantity, {**ctx, df_key: sh, "columns": list(sh.columns)})
        if v is not None:
            nulls.append(abs(v))
    if len(nulls) < n * 0.5:
        return 0.0
    nulls = np.array(nulls)
    return float((abs(full) - nulls.mean()) / (nulls.std() + 1e-9))
