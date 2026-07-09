"""Variation-MDL: universal amplifier requiring no task-specific verifier.

Core principle (Compression-as-Cognition, universal form):
  Run weak model K times → responses R1..RK vary.
  Shortest program explaining the variation = signal (what the model stably knows).
  Large residuals = uncertainty → generate probes → recurse.

Why weak > strong here:
  Strong model is overconfident (low variance) → little variation to compress.
  Weak model varies richly → MDL extracts stable core that is robust,
  not a single hallucinated answer.

No task-specific verifier needed: the K responses ARE the observations.
Works on text, code, math, science — any output type.
"""

from __future__ import annotations

import difflib
import gzip
import json
import re
import time
from dataclasses import dataclass, field
from typing import Callable


# ---------------------------------------------------------------------------
# Bit-cost helpers
# ---------------------------------------------------------------------------

def _bits(s: str) -> float:
    """Approximate description length (gzip = practical universal code)."""
    if not s:
        return 0.0
    return len(gzip.compress(s.encode("utf-8", errors="replace"))) * 8.0


def _patch_bits(core: str, response: str) -> float:
    """Bits to encode `response` given `core` (diff as residual)."""
    if core == response:
        return 0.0
    diff = "\n".join(difflib.unified_diff(
        core.splitlines(), response.splitlines(), lineterm="", n=0))
    return _bits(diff) if diff else 0.0


def _mdl_score(core: str, responses: list[str]) -> float:
    """Two-part code: L(core) + sum L(Ri | core)."""
    return _bits(core) + sum(_patch_bits(core, r) for r in responses)


def _baseline(responses: list[str]) -> float:
    """Verbatim storage of all responses (no compression)."""
    return sum(_bits(r) for r in responses)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class VarMDLResult:
    task: str
    core: str                       # MDL-best invariant structure
    patches: list[str]              # per-response residual diffs
    baseline_bits: float
    mdl_bits: float
    compression_ratio: float        # baseline / mdl (> 1 = structure found)
    uncertainty: float              # mean(patch_bits) / bits(core) — high = probes needed
    probe_questions: list[str]      # questions targeting uncertain parts
    rounds: int
    n_responses: int
    trace: list[dict] = field(default_factory=list)
    wall_time_s: float = 0.0


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class VariationMDLEngine:
    """Universal amplifier: weak model × K → strong answer via MDL.

    propose_fn(prompt, k) -> list[str]  — calls the weak model
    meta_fn(prompt) -> str              — calls any model for meta-compression
    """

    def __init__(
        self,
        propose_fn: Callable[[str, int], list[str]],
        meta_fn: Callable[[str], str],
        k_responses: int = 20,
        max_rounds: int = 3,
        compression_threshold: float = 1.3,
    ):
        self.propose = propose_fn
        self.meta = meta_fn
        self.k = k_responses
        self.max_rounds = max_rounds
        self.threshold = compression_threshold

    # ------------------------------------------------------------------
    def run(self, task: str) -> VarMDLResult:
        t0 = time.time()
        trace: list[dict] = []
        all_responses: list[str] = []
        best_core = ""
        best_score = float("inf")
        probe_context = ""
        probe_questions: list[str] = []

        for rnd in range(self.max_rounds):
            # 1. Generate K responses (weak model)
            prompt = self._response_prompt(task, probe_context)
            responses = self.propose(prompt, self.k)
            responses = [r.strip() for r in responses if r.strip()]
            if not responses:
                break
            all_responses.extend(responses)

            # 2. Ask meta-LLM to propose cores (compressed explanations)
            core_candidates = self._propose_cores(task, responses)

            # 3. Select by MDL
            for core in core_candidates:
                score = _mdl_score(core, responses)
                trace.append({
                    "round": rnd + 1,
                    "core_preview": core[:120],
                    "mdl_bits": round(score, 1),
                    "baseline_bits": round(_baseline(responses), 1),
                    "ratio": round(_baseline(responses) / max(score, 1), 3),
                })
                if score < best_score:
                    best_score = score
                    best_core = core

            bl = _baseline(responses)
            ratio = bl / max(best_score, 1)

            # 4. If compressed well enough → done
            if ratio >= self.threshold:
                break

            # 5. Otherwise generate probes targeting residuals
            probe_questions = self._generate_probes(task, best_core, responses)
            probe_context = self._probe_context(probe_questions)

        # Final patches
        patches = [
            "\n".join(difflib.unified_diff(
                best_core.splitlines(), r.splitlines(), lineterm="", n=1))
            for r in (all_responses or [""])
        ]
        bl_total = _baseline(all_responses or [""])
        mdl_total = _mdl_score(best_core, all_responses or [""])
        mean_patch = sum(_patch_bits(best_core, r) for r in (all_responses or [""])) / max(len(all_responses), 1)
        core_bits = _bits(best_core)

        return VarMDLResult(
            task=task,
            core=best_core,
            patches=patches,
            baseline_bits=round(bl_total, 1),
            mdl_bits=round(mdl_total, 1),
            compression_ratio=round(bl_total / max(mdl_total, 1), 3),
            uncertainty=round(mean_patch / max(core_bits, 1), 3),
            probe_questions=probe_questions,
            rounds=rnd + 1,
            n_responses=len(all_responses),
            trace=trace,
            wall_time_s=round(time.time() - t0, 2),
        )

    # ------------------------------------------------------------------
    def _response_prompt(self, task: str, probe_context: str) -> str:
        base = (
            f"Task: {task}\n\n"
            "Give your best answer. Be specific and concrete. "
            "Do not hedge — commit to a specific answer."
        )
        if probe_context:
            base += f"\n\nAdditional context from prior analysis:\n{probe_context}"
        return base

    def _propose_cores(self, task: str, responses: list[str]) -> list[str]:
        """Ask meta-LLM to find the invariant core explaining all responses."""
        sample = responses[:min(len(responses), 8)]
        formatted = "\n\n".join(f"[Response {i+1}]\n{r}" for i, r in enumerate(sample))
        prompt = (
            f"Task: {task}\n\n"
            f"The following {len(sample)} responses were generated independently:\n\n"
            f"{formatted}\n\n"
            "Your job: find the SHORTEST answer that captures what ALL (or most) responses "
            "agree on — the invariant core. Ignore idiosyncratic details. "
            "Output ONLY the core answer, nothing else. "
            "The shorter and more precise, the better."
        )
        core = self.meta(prompt).strip()
        # Also include the most common response as a candidate
        by_freq = sorted(set(responses), key=lambda r: -responses.count(r))
        candidates = [core] + by_freq[:2]
        return [c for c in candidates if c]

    def _generate_probes(self, task: str, core: str, responses: list[str]) -> list[str]:
        """Generate questions that would resolve residual uncertainty."""
        # Find high-variance parts: lines that differ across responses
        all_lines = []
        for r in responses:
            all_lines.extend(r.splitlines())
        line_counts: dict[str, int] = {}
        for l in all_lines:
            line_counts[l.strip()] = line_counts.get(l.strip(), 0) + 1
        # Lines that appear in fewer than half the responses = uncertain
        uncertain = [l for l, c in line_counts.items()
                     if c < len(responses) / 2 and len(l) > 10][:5]

        if not uncertain:
            return []

        uncertain_text = "\n".join(uncertain[:5])
        prompt = (
            f"Task: {task}\n\n"
            f"Current best answer: {core}\n\n"
            f"Uncertain parts (appear in only some responses):\n{uncertain_text}\n\n"
            "Generate 2-3 specific, factual questions whose answers would resolve "
            "this uncertainty. Output one question per line, nothing else."
        )
        raw = self.meta(prompt).strip()
        questions = [q.strip() for q in raw.splitlines() if q.strip() and "?" in q]
        return questions[:3]

    def _probe_context(self, questions: list[str]) -> str:
        if not questions:
            return ""
        return "Focus on answering these specific questions:\n" + "\n".join(
            f"- {q}" for q in questions)


# ---------------------------------------------------------------------------
# Factory: connect to existing LLM infra
# ---------------------------------------------------------------------------

def make_variation_engine(
    weak_model: str = "openai/gpt-4o-mini",
    meta_model: str = "openai/gpt-4o-mini",
    k: int = 20,
    max_rounds: int = 3,
    client=None,
) -> VariationMDLEngine:
    """Wire up to the project's call_llm infra."""
    import sys
    from pathlib import Path
    _proj = Path(__file__).resolve().parent.parent.parent
    for _p in (_proj, _proj / "ultrahorizon_repo"):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

    from mars.agents.base import call_llm, make_openai_client
    _client = client or make_openai_client()

    def propose_fn(prompt: str, k: int) -> list[str]:
        # K independent calls at temperature=0.8 for diversity
        results = []
        for _ in range(k):
            try:
                r = call_llm(_client, model=weak_model,
                             system="You are a precise scientific analyst.",
                             user=prompt, max_tokens=600, temperature=0.85)
                if r:
                    results.append(r.strip())
            except Exception:
                pass
        return results

    def meta_fn(prompt: str) -> str:
        try:
            return call_llm(_client, model=meta_model,
                            system="You are a precise scientific analyst. Extract invariant structure.",
                            user=prompt, max_tokens=800, temperature=0.2)
        except Exception:
            return ""

    return VariationMDLEngine(propose_fn, meta_fn, k_responses=k, max_rounds=max_rounds)
