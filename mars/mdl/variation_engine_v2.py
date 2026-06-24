"""Variation-MDL v2: Semantic MDL + Information-Gain Probe Selection.

Fixes from v1:
  - Structured JSON output → field-level MDL (not syntactic gzip+diff)
  - compression_ratio > 1 correctly when structure is found
  - Probe selection maximizes expected information gain (optimal experimental design)
  - Routes probes to executable verifiers when available

Core theory (Compression-as-Cognition, universal form):
  K responses → structured fields → MDL over fields → signal vs noise
  High-residual fields = uncertainty → probe targets max information gain
  Probe → verifier → new bits → recurse

Why weak(K) > strong(K=1):
  Strong model: low entropy → single confident answer → no uncertainty signal
  Weak model: high entropy → field-level variation → MDL extracts invariant core
  As K → ∞: MDL core converges to truth at rate O(H(task|weak_model) / K)
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------

STRUCTURED_PROMPT = """\
Task: {task}

{context}

Answer this task. Output ONLY valid JSON with these fields:
{{
  "mechanism": "<core rule or mechanism in one sentence>",
  "formula": "<mathematical or logical formula, or null>",
  "threshold": "<key threshold or boundary value, or null>",
  "direction": "<sign of effect: positive/negative/both/null>",
  "conditions": ["<condition 1>", "<condition 2>"],
  "answer": "<final answer in one clear sentence>"
}}
Be specific and commit to concrete values. No hedging."""


# ---------------------------------------------------------------------------
# Field-level MDL scoring
# ---------------------------------------------------------------------------

def _str_bits(s: str) -> float:
    """Bits to store a string (entropy-coded ASCII)."""
    if not s:
        return 0.0
    return len(s) * math.log2(96)   # 96 printable ASCII


def _field_residual_bits(core_val: Any, resp_val: Any) -> float:
    """Bits to encode resp_val given core_val as context."""
    if core_val == resp_val:
        return 0.0
    if isinstance(core_val, (int, float)) and isinstance(resp_val, (int, float)):
        # numeric: relative log-error in bits
        if core_val == 0:
            return _str_bits(str(resp_val))
        rel = abs(resp_val - core_val) / max(abs(core_val), 1e-9)
        return max(0.0, math.log2(rel + 1) * 8)
    if isinstance(core_val, str) and isinstance(resp_val, str):
        # string: normalized edit distance
        longer = max(len(core_val), len(resp_val), 1)
        edits = _levenshtein_approx(core_val, resp_val)
        return (edits / longer) * _str_bits(resp_val)
    if isinstance(core_val, list) and isinstance(resp_val, list):
        # list: symmetric difference size
        s1, s2 = set(str(x) for x in core_val), set(str(x) for x in resp_val)
        diff = len(s1.symmetric_difference(s2))
        return diff * 4.0
    # fallback: full cost
    return _str_bits(str(resp_val))


def _levenshtein_approx(a: str, b: str, cap: int = 200) -> int:
    """Fast approximation: only compare first `cap` chars."""
    a, b = a[:cap], b[:cap]
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            curr.append(min(prev[j + 1] + 1, curr[-1] + 1,
                            prev[j] + (0 if ca == cb else 1)))
        prev = curr
    return prev[-1]


FIELDS = ["mechanism", "formula", "threshold", "direction", "conditions", "answer"]


def _mdl_score_structured(core: dict, responses: list[dict]) -> float:
    """Two-part code over structured fields: L(core) + sum L(Ri | core)."""
    # Program bits: describe the core
    prog_bits = sum(_str_bits(str(core.get(f, ""))) for f in FIELDS)
    # Residual bits: per-response field deviations
    res_bits = 0.0
    for resp in responses:
        for f in FIELDS:
            res_bits += _field_residual_bits(core.get(f), resp.get(f))
    return prog_bits + res_bits


def _baseline_structured(responses: list[dict]) -> float:
    """Verbatim storage: store each response fully."""
    return sum(
        sum(_str_bits(str(r.get(f, ""))) for f in FIELDS)
        for r in responses
    )


def _extract_core_modal(responses: list[dict]) -> dict:
    """Fallback: modal consensus per field."""
    core: dict = {}
    for f in FIELDS:
        vals = [r.get(f) for r in responses if r.get(f) is not None]
        if not vals:
            core[f] = None
            continue
        if isinstance(vals[0], list):
            flat = [str(x) for v in vals for x in v]
            counts: dict[str, int] = {}
            for x in flat:
                counts[x] = counts.get(x, 0) + 1
            thresh = len(vals) * 0.5
            core[f] = [k for k, c in counts.items() if c >= thresh]
        else:
            counts2: dict[str, int] = {}
            for v in vals:
                counts2[str(v)] = counts2.get(str(v), 0) + 1
            core[f] = max(counts2, key=counts2.__getitem__)
    return core


def _extract_core(responses: list[dict], meta_fn=None, task: str = "") -> dict:
    """MDL-optimal core: propose candidates, select by minimum description length.

    Generates candidate cores from:
      1. Each unique response (verbatim candidate)
      2. Modal consensus across fields
      3. Meta-LLM synthesised core (if meta_fn provided)

    Then selects the candidate minimising L(core) + sum_i L(Ri | core).
    This is the correct MDL aggregation — not just mode-counting.
    """
    candidates: list[dict] = []

    # Candidate 1: modal consensus (fast baseline)
    candidates.append(_extract_core_modal(responses))

    # Candidates 2+: unique individual responses as-is
    seen: set[str] = set()
    for r in responses:
        key = str(r.get("answer", "")) + str(r.get("mechanism", ""))
        if key not in seen:
            seen.add(key)
            candidates.append(r)
        if len(candidates) > 12:
            break

    # Candidate N: meta-LLM synthesised core
    if meta_fn and task:
        sample = responses[:min(len(responses), 6)]
        formatted = "\n".join(
            f"Response {i+1}: mechanism={r.get('mechanism','')} | "
            f"threshold={r.get('threshold','')} | answer={r.get('answer','')}"
            for i, r in enumerate(sample)
        )
        prompt = (
            f"Task: {task}\n\n"
            f"Here are {len(sample)} structured responses:\n{formatted}\n\n"
            "Synthesise the SHORTEST correct answer that is consistent with "
            "the majority of responses. Output ONLY valid JSON with fields: "
            "mechanism, formula, threshold, direction, conditions, answer."
        )
        try:
            raw = meta_fn(prompt)
            synth = _parse_json(raw)
            if synth:
                candidates.append(synth)
        except Exception:
            pass

    # Select by MDL: minimum two-part code length
    best_core = candidates[0]
    best_score = _mdl_score_structured(candidates[0], responses)
    for c in candidates[1:]:
        s = _mdl_score_structured(c, responses)
        if s < best_score:
            best_score = s
            best_core = c
    return best_core


# ---------------------------------------------------------------------------
# Information-gain probe selection
# ---------------------------------------------------------------------------

def _field_entropy(responses: list[dict], field: str) -> float:
    """Shannon entropy of a field's distribution across responses."""
    vals = [str(r.get(field, "")) for r in responses]
    counts: dict[str, int] = {}
    for v in vals:
        counts[v] = counts.get(v, 0) + 1
    n = len(vals)
    if n == 0:
        return 0.0
    return -sum((c / n) * math.log2(c / n) for c in counts.values() if c > 0)


def select_probe_fields(responses: list[dict], core: dict, top_k: int = 2) -> list[str]:
    """Fields with highest residual entropy = maximum information gain target."""
    entropies = []
    for f in FIELDS:
        # Residual entropy: entropy of (resp[f] - core[f]) distribution
        residuals = [str(r.get(f, "")) for r in responses
                     if str(r.get(f, "")) != str(core.get(f, ""))]
        if len(residuals) < 2:
            entropies.append((f, 0.0))
            continue
        counts: dict[str, int] = {}
        for v in residuals:
            counts[v] = counts.get(v, 0) + 1
        n = len(residuals)
        h = -sum((c / n) * math.log2(c / n) for c in counts.values() if c > 0)
        entropies.append((f, h * len(residuals)))  # weight by frequency
    entropies.sort(key=lambda x: -x[1])
    return [f for f, _ in entropies[:top_k] if _ > 0]


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class VarMDLv2Result:
    task: str
    core: dict                      # MDL-optimal structured answer
    answer: str                     # core["answer"]
    baseline_bits: float
    mdl_bits: float
    compression_ratio: float        # > 1.3 = real structure found
    uncertain_fields: list[str]     # fields with high residual entropy
    probe_questions: list[str]      # targeted at uncertain fields
    rounds: int
    n_responses: int
    trace: list[dict] = field(default_factory=list)
    wall_time_s: float = 0.0


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class VariationMDLEngineV2:
    """Universal amplifier v2: semantic MDL over structured responses.

    propose_fn(prompt, k) -> list[str]    — weak model, K samples
    meta_fn(prompt) -> str                — any model for probe generation
    verifier_fn(probe, field) -> str|None — optional: executable verifier
    """

    def __init__(
        self,
        propose_fn: Callable[[str, int], list[str]],
        meta_fn: Callable[[str], str],
        verifier_fn: Optional[Callable[[str, str], Optional[str]]] = None,
        k_responses: int = 20,
        max_rounds: int = 3,
        compression_threshold: float = 1.3,
    ):
        self.propose = propose_fn
        self.meta = meta_fn
        self.verify = verifier_fn
        self.k = k_responses
        self.max_rounds = max_rounds
        self.threshold = compression_threshold

    def run(self, task: str) -> VarMDLv2Result:
        t0 = time.time()
        trace: list[dict] = []
        all_structs: list[dict] = []
        context = ""
        probe_questions: list[str] = []
        best_core: dict = {}
        best_ratio = 0.0

        for rnd in range(self.max_rounds):
            # 1. Generate K structured responses
            prompt = STRUCTURED_PROMPT.format(task=task, context=context)
            raw_responses = self.propose(prompt, self.k)
            structs = [_parse_json(r) for r in raw_responses]
            structs = [s for s in structs if s]
            if not structs:
                break
            all_structs.extend(structs)

            # 2. Extract MDL-optimal core: propose candidates, select by MDL
            core = _extract_core(all_structs, meta_fn=self.meta, task=task)

            # 3. Score
            bl = _baseline_structured(all_structs)
            mdl = _mdl_score_structured(core, all_structs)
            ratio = bl / max(mdl, 1.0)

            trace.append({
                "round": rnd + 1,
                "n_responses": len(all_structs),
                "baseline_bits": round(bl, 1),
                "mdl_bits": round(mdl, 1),
                "compression_ratio": round(ratio, 3),
                "core_answer": (core.get("answer") or "")[:100],
                "core_mechanism": (core.get("mechanism") or "")[:80],
            })

            if ratio > best_ratio:
                best_ratio = ratio
                best_core = core

            # 4. If well compressed → done
            if ratio >= self.threshold:
                break

            # 5. Find high-entropy fields → generate probes
            uncertain = select_probe_fields(all_structs, core, top_k=2)
            probes = self._generate_probes(task, core, uncertain, all_structs)
            probe_questions = probes

            # 6. Route probes to verifier if available
            resolved = []
            for probe_q, probe_field in zip(probes, uncertain):
                if self.verify:
                    answer = self.verify(probe_q, probe_field)
                    if answer:
                        resolved.append(f"Q: {probe_q}\nA: {answer}")
                else:
                    resolved.append(f"Focus on resolving: {probe_q}")
            context = "\n".join(resolved)

        bl_final = _baseline_structured(all_structs)
        mdl_final = _mdl_score_structured(best_core, all_structs)
        uncertain_final = select_probe_fields(all_structs, best_core, top_k=3)

        return VarMDLv2Result(
            task=task,
            core=best_core,
            answer=str(best_core.get("answer") or best_core.get("mechanism") or ""),
            baseline_bits=round(bl_final, 1),
            mdl_bits=round(mdl_final, 1),
            compression_ratio=round(bl_final / max(mdl_final, 1.0), 3),
            uncertain_fields=uncertain_final,
            probe_questions=probe_questions,
            rounds=rnd + 1,
            n_responses=len(all_structs),
            trace=trace,
            wall_time_s=round(time.time() - t0, 2),
        )

    def _generate_probes(self, task: str, core: dict,
                         uncertain_fields: list[str],
                         responses: list[dict]) -> list[str]:
        if not uncertain_fields:
            return []
        field_examples = {}
        for f in uncertain_fields:
            vals = list({str(r.get(f, "")) for r in responses if r.get(f)})[:4]
            field_examples[f] = vals

        prompt = (
            f"Task: {task}\n\n"
            f"Current best answer: {json.dumps(core, indent=2)}\n\n"
            f"The following fields are uncertain (high variation across responses):\n"
            + "\n".join(
                f"  {f}: candidates = {field_examples.get(f, [])}"
                for f in uncertain_fields
            )
            + "\n\nGenerate one specific, answerable question per uncertain field "
            "that would definitively resolve which value is correct. "
            "Output one question per line."
        )
        raw = self.meta(prompt).strip()
        return [q.strip() for q in raw.splitlines()
                if q.strip() and len(q.strip()) > 10][:len(uncertain_fields)]


# ---------------------------------------------------------------------------
# JSON parser (robust)
# ---------------------------------------------------------------------------

def _parse_json(text: str) -> Optional[dict]:
    text = text.strip()
    # Extract JSON block
    m = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if m:
        text = m.group(0)
    try:
        d = json.loads(text)
        if isinstance(d, dict):
            return {f: d.get(f) for f in FIELDS}
    except Exception:
        pass
    # Try to extract fields manually
    result: dict = {}
    for f in FIELDS:
        m2 = re.search(rf'"{f}"\s*:\s*"([^"]*)"', text)
        if m2:
            result[f] = m2.group(1)
    return result if result else None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_v2_engine(
    weak_model: str = "openai/gpt-4o-mini",
    meta_model: str = "openai/gpt-4o-mini",
    k: int = 20,
    max_rounds: int = 3,
    verifier_fn=None,
    client=None,
) -> VariationMDLEngineV2:
    import sys
    from pathlib import Path
    _proj = Path(__file__).resolve().parent.parent.parent
    for _p in (_proj, _proj / "ultrahorizon_repo"):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
    from mars.agents.base import call_llm, make_openai_client
    _client = client or make_openai_client()

    def propose_fn(prompt: str, k: int) -> list[str]:
        results = []
        for _ in range(k):
            try:
                r = call_llm(_client, model=weak_model,
                             system="You are a precise scientific analyst. Always output valid JSON.",
                             user=prompt, max_tokens=500, temperature=0.85)
                if r:
                    results.append(r.strip())
            except Exception:
                pass
        return results

    def meta_fn(prompt: str) -> str:
        try:
            return call_llm(_client, model=meta_model,
                            system="You are a precise scientific analyst.",
                            user=prompt, max_tokens=400, temperature=0.2)
        except Exception:
            return ""

    return VariationMDLEngineV2(propose_fn, meta_fn,
                                verifier_fn=verifier_fn,
                                k_responses=k, max_rounds=max_rounds)
