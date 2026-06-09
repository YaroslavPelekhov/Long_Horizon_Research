"""Universal CPI — ONE autonomous hypothesis-induction engine for all benchmarks.

The thesis: a single engine discovers refutable hypotheses across heterogeneous
benchmarks WITHOUT any benchmark-specific answer being written by a human.

The engine never sees the scoring rubric or the ground-truth answer during
search. It only sees the OBSERVABLE INTERFACE and OBSERVATIONS collected from
the environment, then:

  1. asks an LLM to PROPOSE candidate analyzer programs (typed to the interface)
  2. SANDBOX-validates them (AST safety + smoke execution)
  3. SCORES each program by REFUTATION on held-out observations
  4. SELECTS winners by MDL (loss + complexity)
  5. RENDERS a benchmark-facing report from the winning programs only

The ONLY benchmark-specific code lives in a thin CPIAdapter that says:
  - what the observable interface looks like (schema, NOT answers)
  - how to collect observations from the environment
  - how to execute a candidate program on one observation
  - how to measure prediction loss against an observation
  - how to render winners into the benchmark's expected output format

Crucially, the adapter contains NO domain answers. The hypotheses are
discovered, not printed. This is what makes the system autonomous AND universal.
"""

from __future__ import annotations

import ast
import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from mars.agents.base import call_llm, make_openai_client


# ===========================================================================
# Core data types
# ===========================================================================

@dataclass
class Observation:
    """One piece of evidence from the environment.

    `inputs` is whatever the candidate program receives.
    `target` is the observed outcome the program must predict.
    `context` carries auxiliary signals (step number, schema, metadata).
    """
    inputs: Any
    target: Any
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class HypothesisProgram:
    """An executable, refutable hypothesis proposed by the LLM."""
    name: str
    description: str
    code: str
    fn: Callable
    complexity: float = 2.0
    tags: tuple[str, ...] = ("proposed",)


@dataclass
class ProgramScore:
    name: str
    loss_mean: float
    exact_rate: float
    mdl_score: float
    n_scored: int
    complexity: float


@dataclass
class CPIResult:
    benchmark: str
    n_observations: int
    n_proposed: int
    n_valid: int
    winners: list[tuple[HypothesisProgram, ProgramScore]]
    report: str
    proposals_raw: list[dict]
    errors: list[str]
    wall_time_s: float
    rounds: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "n_observations": self.n_observations,
            "n_proposed": self.n_proposed,
            "n_valid": self.n_valid,
            "rounds": self.rounds,
            "wall_time_s": self.wall_time_s,
            "winners": [
                {
                    "name": h.name,
                    "description": h.description,
                    "loss_mean": s.loss_mean,
                    "exact_rate": s.exact_rate,
                    "mdl_score": s.mdl_score,
                    "complexity": h.complexity,
                    "code": h.code,
                }
                for h, s in self.winners
            ],
            "errors": self.errors[:10],
        }


# ===========================================================================
# Sandbox (shared by all benchmarks)
# ===========================================================================

_SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs, "all": all, "any": any, "bool": bool, "chr": chr, "dict": dict,
    "divmod": divmod, "enumerate": enumerate, "filter": filter, "float": float,
    "int": int, "isinstance": isinstance, "len": len, "list": list, "map": map,
    "max": max, "min": min, "ord": ord, "pow": pow, "range": range, "reversed": reversed,
    "round": round, "set": set, "sorted": sorted, "str": str, "sum": sum,
    "tuple": tuple, "zip": zip, "True": True, "False": False, "None": None,
}

_BANNED_NAMES = {"eval", "exec", "open", "compile", "__import__", "breakpoint",
                 "input", "globals", "locals", "vars", "getattr", "setattr", "delattr"}
_BANNED_ATTRS = {"__class__", "__globals__", "__code__", "__bases__", "__subclasses__",
                 "__mro__", "__dict__", "__builtins__"}


def _ast_safe(code: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            # Allow a small whitelist of math/stats imports
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            else:
                names = [node.module or ""]
            for n in names:
                if n.split(".")[0] not in ("math", "statistics", "itertools"):
                    return False, f"import not allowed: {n}"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _BANNED_NAMES:
                return False, f"unsafe call: {node.func.id}"
        if isinstance(node, ast.Attribute) and node.attr in _BANNED_ATTRS:
            return False, f"unsafe attribute: {node.attr}"
    return True, ""


def _sandbox_compile(
    code: str,
    extra_globals: dict[str, Any] | None = None,
) -> tuple[bool, str, Any]:
    ok, err = _ast_safe(code)
    if not ok:
        return False, err, None
    ns: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS}
    # Allow whitelisted modules
    import math as _math
    import statistics as _stats
    import itertools as _itertools
    ns["math"] = _math
    ns["statistics"] = _stats
    ns["itertools"] = _itertools
    if extra_globals:
        ns.update(extra_globals)
    try:
        exec(compile(code, "<universal_cpi>", "exec"), ns)
    except Exception as e:
        return False, f"ExecError: {e}", None
    fn = next(
        (v for k, v in ns.items()
         if callable(v) and not isinstance(v, type)
         and not k.startswith("_") and k not in ("math", "statistics", "itertools")),
        None,
    )
    if fn is None:
        return False, "no callable found", None
    return True, "", fn


# ===========================================================================
# Benchmark adapter interface — the ONLY place benchmark knowledge lives.
# Contains NO answers, only mechanics.
# ===========================================================================

class CPIAdapter(ABC):
    """Thin per-benchmark binding. NO domain answers allowed here."""

    name: str = "abstract"

    @abstractmethod
    def interface_description(self) -> str:
        """Schema shown to the LLM. Types and available variables ONLY.
        Must NOT contain the answer or the scoring rubric."""

    @abstractmethod
    def collect_observations(self) -> list[Observation]:
        """Gather observations from the environment."""

    @abstractmethod
    def execute(self, program: HypothesisProgram, obs: Observation) -> Any:
        """Run a candidate program on one observation, return its prediction."""

    @abstractmethod
    def loss(self, prediction: Any, obs: Observation) -> float:
        """Normalized loss in [0,1]: 0 = perfect, 1 = wrong/failed."""

    @abstractmethod
    def render_report(
        self,
        winners: list[tuple[HypothesisProgram, ProgramScore]],
        observations: list[Observation],
    ) -> str:
        """Translate winning programs into the benchmark's expected output.
        Generic verbalization of DISCOVERED programs, not canned answers."""

    # Optional: extra sandbox globals (e.g. pandas, numpy)
    def sandbox_globals(self) -> dict[str, Any]:
        return {}

    # Optional: fit free constants in a proposed program against training
    # observations (e.g. the multiplicative constant in a physical law).
    # This is data-driven calibration, NOT an answer. Default: no-op.
    def calibrate(
        self,
        program: "HypothesisProgram",
        train_observations: list["Observation"],
    ) -> "HypothesisProgram":
        return program

    # Optional: signal hints extracted automatically from observations.
    # These are DATA-DERIVED facts (e.g. "output length = 2x input"),
    # not answers. Default: nothing.
    def auto_signals(self, observations: list[Observation]) -> str:
        return ""

    # The proposal function-signature contract for the LLM.
    @abstractmethod
    def signature_hint(self) -> str:
        """The exact Python signature the proposed function must have."""


# ===========================================================================
# The universal engine
# ===========================================================================

class UniversalCPI:
    """ONE engine. Propose → validate → refute → select → render."""

    def __init__(
        self,
        *,
        model: str = "openai/gpt-4o",
        n_proposals: int = 16,
        holdout_frac: float = 0.4,
        complexity_weight: float = 0.01,
        max_rounds: int = 2,
        temperature: float = 0.8,
    ):
        self.model = model
        self.n_proposals = n_proposals
        self.holdout_frac = holdout_frac
        self.complexity_weight = complexity_weight
        self.max_rounds = max_rounds
        self.temperature = temperature
        self._client = None

    def _client_lazy(self):
        if self._client is None:
            self._client = make_openai_client()
        return self._client

    # ----- Proposal -------------------------------------------------------
    def _build_prompt(
        self,
        adapter: CPIAdapter,
        observations: list[Observation],
        failure_context: str = "",
    ) -> str:
        examples = []
        for i, obs in enumerate(observations[:8]):
            examples.append({
                "ex": i + 1,
                "inputs": _truncate_repr(obs.inputs),
                "context": {k: _truncate_repr(v) for k, v in obs.context.items()
                            if not k.startswith("__")},
                "target": _truncate_repr(obs.target),
            })
        signals = adapter.auto_signals(observations)
        prompt = f"""{adapter.interface_description()}

FUNCTION SIGNATURE (your functions MUST match exactly):
{adapter.signature_hint()}

OBSERVED EVIDENCE (inputs + context → target):
{json.dumps(examples, indent=2, ensure_ascii=False, default=str)[:3500]}
{signals}
{failure_context}

Propose {self.n_proposals} diverse candidate functions, each a DIFFERENT
mechanistic hypothesis for how inputs map to target. Some will be wrong — that
is expected; they will be scored by refutation on held-out evidence.

Return ONLY JSON:
{{
  "programs": [
    {{
      "name": "snake_case_name",
      "description": "one-line mechanism description",
      "complexity": <1-5 integer>,
      "code": "def NAME(...):\\n    ..."
    }}
  ]
}}"""
        return prompt

    def _propose(
        self,
        adapter: CPIAdapter,
        observations: list[Observation],
        failure_context: str = "",
    ) -> tuple[list[HypothesisProgram], list[dict], list[str]]:
        prompt = self._build_prompt(adapter, observations, failure_context)
        raw = call_llm(
            self._client_lazy(),
            model=self.model,
            system=(
                "You are an autonomous scientific hypothesis engine. Propose diverse "
                "executable hypotheses to explain observed evidence. Return only valid JSON."
            ),
            user=prompt,
            max_tokens=4500,
            temperature=self.temperature,
        )
        proposals = _parse_programs(raw)
        programs: list[HypothesisProgram] = []
        errors: list[str] = []
        sg = adapter.sandbox_globals()
        for p in proposals:
            if not isinstance(p, dict):
                continue
            code = (p.get("code") or "").strip()
            if not code:
                errors.append(f"{p.get('name','?')}: no code")
                continue
            ok, err, fn = _sandbox_compile(code, sg)
            if not ok:
                errors.append(f"{p.get('name','?')}: {err}")
                continue
            programs.append(HypothesisProgram(
                name=p.get("name", f"prog_{len(programs)}"),
                description=p.get("description", ""),
                code=code,
                fn=fn,
                complexity=float(p.get("complexity", 2.0)),
            ))
        return programs, proposals, errors

    # ----- Scoring (refutation on held-out) -------------------------------
    def _score(
        self,
        program: HypothesisProgram,
        observations: list[Observation],
        adapter: CPIAdapter,
    ) -> ProgramScore:
        losses: list[float] = []
        exact = 0
        for obs in observations:
            try:
                pred = adapter.execute(program, obs)
                l = adapter.loss(pred, obs)
                if l != l or l < 0:  # NaN guard
                    l = 1.0
                l = min(1.0, max(0.0, l))
            except Exception:
                l = 1.0
            exact += int(l == 0.0)
            losses.append(l)
        mean = sum(losses) / len(losses) if losses else 1.0
        return ProgramScore(
            name=program.name,
            loss_mean=mean,
            exact_rate=exact / len(losses) if losses else 0.0,
            mdl_score=mean + self.complexity_weight * program.complexity,
            n_scored=len(losses),
            complexity=program.complexity,
        )

    def _rank(
        self,
        programs: list[HypothesisProgram],
        observations: list[Observation],
        adapter: CPIAdapter,
    ) -> list[tuple[HypothesisProgram, ProgramScore]]:
        ranked = [(p, self._score(p, observations, adapter)) for p in programs]
        ranked.sort(key=lambda x: (x[1].mdl_score, x[1].loss_mean, x[0].complexity))
        return ranked

    # ----- Main loop ------------------------------------------------------
    def run(self, adapter: CPIAdapter) -> CPIResult:
        t0 = time.time()
        observations = adapter.collect_observations()
        if not observations:
            return CPIResult(adapter.name, 0, 0, 0, [], "", [], ["no observations"], 0.0)

        # Train/holdout split for refutation
        n_hold = max(1, int(len(observations) * self.holdout_frac))
        train_obs = observations[:-n_hold] or observations
        hold_obs = observations[-n_hold:] or observations

        all_programs: list[HypothesisProgram] = []
        all_proposals: list[dict] = []
        all_errors: list[str] = []
        failure_context = ""
        rounds = 0

        for rnd in range(self.max_rounds):
            rounds = rnd + 1
            programs, proposals, errors = self._propose(adapter, train_obs, failure_context)
            # Calibrate free constants against training observations (data-driven)
            programs = [adapter.calibrate(p, train_obs) for p in programs]
            all_programs.extend(programs)
            all_proposals.extend(proposals)
            all_errors.extend(errors)

            if not all_programs:
                failure_context = "\nPREVIOUS ROUND: no valid programs compiled. Use only the allowed builtins and the exact signature."
                continue

            # Score on held-out (refutation)
            ranked = self._rank(all_programs, hold_obs, adapter)
            best_loss = ranked[0][1].loss_mean if ranked else 1.0

            if best_loss == 0.0:
                break  # perfect fit, stop early

            # Build failure context for next round from the best imperfect program
            best_prog, best_score = ranked[0]
            failure_context = (
                f"\nPREVIOUS ROUND best program '{best_prog.name}' achieved "
                f"loss={best_score.loss_mean:.3f} (not perfect). Propose NEW mechanisms "
                f"that differ structurally — consider interactions, conditionals on "
                f"context variables, history, and compositions you have not tried."
            )

        # Final ranking on ALL observations (train+holdout) for the report
        ranked_full = self._rank(all_programs, observations, adapter) if all_programs else []
        winners = ranked_full[:5]

        report = adapter.render_report(winners, observations) if winners else ""

        return CPIResult(
            benchmark=adapter.name,
            n_observations=len(observations),
            n_proposed=len(all_proposals),
            n_valid=len(all_programs),
            winners=winners,
            report=report,
            proposals_raw=all_proposals,
            errors=all_errors,
            wall_time_s=time.time() - t0,
            rounds=rounds,
        )


# ===========================================================================
# Helpers
# ===========================================================================

def _truncate_repr(value: Any, limit: int = 200) -> Any:
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    s = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    return s if len(s) <= limit else s[:limit] + "…"


def _parse_programs(raw: str) -> list[dict]:
    if not raw:
        return []
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).lstrip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            for key in ("programs", "primitives", "functions", "hypotheses"):
                if isinstance(obj.get(key), list):
                    return obj[key]
        if isinstance(obj, list):
            return obj
    except Exception:
        pass
    m = re.search(r'\[\s*\{.*\}\s*\]', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    m2 = re.search(r'\{.*\}', text, re.DOTALL)
    if m2:
        try:
            obj = json.loads(m2.group())
            if isinstance(obj, dict):
                for key in ("programs", "primitives", "functions", "hypotheses"):
                    if isinstance(obj.get(key), list):
                        return obj[key]
        except Exception:
            pass
    return []
