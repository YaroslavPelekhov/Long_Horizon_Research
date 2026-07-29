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
import hashlib
import json
import math
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from mars.agents.base import call_llm, make_openai_client
from mars.skills.residual_class_ledger import ResidualClassLedger
from mars.skills.residual_kernel import ResidualKernel
from mars.skills.self_layer_registry import SelfLayerRegistry, validate_self_layer_source
from mars.skills.self_module_registry import SelfModuleRegistry
from mars.skills.typed_operator_plan import (
    admissible_families,
    compile_typed_operator_plan,
)

_ZERO_LOSS_EPS = 1e-9


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
    supermetrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "n_observations": self.n_observations,
            "n_proposed": self.n_proposed,
            "n_valid": self.n_valid,
            "rounds": self.rounds,
            "wall_time_s": self.wall_time_s,
            "report": self.report,
            "winners": [
                {
                    "name": h.name,
                    "description": h.description,
                    "loss_mean": s.loss_mean,
                    "exact_rate": s.exact_rate,
                    "mdl_score": s.mdl_score,
                    "complexity": h.complexity,
                    "code": h.code,
                    "tags": list(h.tags),
                }
                for h, s in self.winners
            ],
            "proposals_raw": self.proposals_raw,
            "errors": self.errors[:10],
            "supermetrics": self.supermetrics,
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


def _sandbox_compile_named(
    code: str,
    fn_name: str,
    extra_globals: dict[str, Any] | None = None,
) -> tuple[bool, str, Any]:
    ok, err = _ast_safe(code)
    if not ok:
        return False, err, None
    ns: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS}
    import math as _math
    import statistics as _stats
    import itertools as _itertools
    ns["math"] = _math
    ns["statistics"] = _stats
    ns["itertools"] = _itertools
    if extra_globals:
        ns.update(extra_globals)
    try:
        exec(compile(code, "<universal_cpi_named>", "exec"), ns)
    except Exception as e:
        return False, f"ExecError: {e}", None
    fn = ns.get(fn_name)
    if not callable(fn):
        return False, f"required callable not found: {fn_name}", None
    return True, "", fn


def _angle_like_key(key: str) -> bool:
    low = str(key).lower()
    return any(token in low for token in ("angle", "theta", "degree"))


def _uninformative_angle_numeric_task(
    adapter: "CPIAdapter",
    observations: list["Observation"],
) -> bool:
    if "inputs: dict" not in adapter.signature_hint() or len(observations) < 3:
        return False
    keys: list[str] = []
    targets: list[float] = []
    input_rows: list[tuple[float, ...]] = []
    for obs in observations:
        if not isinstance(obs.inputs, dict):
            return False
        numeric_row: list[float] = []
        for key, value in obs.inputs.items():
            try:
                numeric_row.append(float(value))
            except Exception:
                continue
            if str(key) not in keys:
                keys.append(str(key))
        try:
            targets.append(float(obs.target))
        except Exception:
            return False
        input_rows.append(tuple(numeric_row))
    if not any(_angle_like_key(key) for key in keys):
        return False
    if len([key for key in keys if not _angle_like_key(key)]) < 2:
        return False
    if max(targets) - min(targets) > 1e-9:
        return False
    return len(set(input_rows)) >= 3


def _program_is_degenerate_constant(
    program: "HypothesisProgram",
    observations: list["Observation"],
) -> bool:
    values: list[float] = []
    for obs in observations[:12]:
        try:
            values.append(float(program.fn(obs.inputs)))
        except Exception:
            return False
    if not values:
        return False
    spread = max(values) - min(values)
    scale = max(1.0, max(abs(v) for v in values))
    return spread <= 1e-5 * scale


def _ordered_ratio_prior_bonus(name: str) -> float:
    match = re.search(r"(?:^|_)([a-zA-Z]+)(\d+)_over_([a-zA-Z]+)(\d+)(?:_|$)", str(name))
    if not match:
        return 0.0
    left_stem, left_idx, right_stem, right_idx = match.groups()
    if left_stem != right_stem:
        return 0.0
    try:
        li = int(left_idx)
        ri = int(right_idx)
    except Exception:
        return 0.0
    if li < ri:
        return 0.08
    if li > ri:
        return -0.08
    return 0.0


def _rename_first_function_source(code: str, new_name: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            node.name = new_name
            ast.fix_missing_locations(tree)
            return ast.unparse(tree)
    return None


def _compose_transform_sources(
    left_code: str,
    right_code: str,
    *,
    left_name: str = "_seed_left",
    right_name: str = "_seed_right",
) -> str | None:
    """Compose two transform programs with signature (current, context)."""

    left = _rename_first_function_source(left_code, left_name)
    right = _rename_first_function_source(right_code, right_name)
    if not left or not right:
        return None
    return (
        f"{left}\n\n"
        f"{right}\n\n"
        "def rule(current: str, context: dict) -> str:\n"
        f"    return {right_name}({left_name}(current, context), context)\n"
    )


def _branch_transform_sources(
    true_code: str,
    false_code: str,
    *,
    predicate_expr: str,
    true_name: str = "_seed_true",
    false_name: str = "_seed_false",
    helper_source: str = "",
) -> str | None:
    """Build a conditional transform graph from two programs.

    The predicate is adapter-provided from observable interface fields only.
    It is still sandbox-compiled before use, so unsafe predicates or seed code
    are rejected by the same AST gate as model-proposed programs.
    """

    true_src = _rename_first_function_source(true_code, true_name)
    false_src = _rename_first_function_source(false_code, false_name)
    if not true_src or not false_src:
        return None
    helper = helper_source.strip()
    helper_block = f"{helper}\n\n" if helper else ""
    return (
        f"{helper_block}"
        f"{true_src}\n\n"
        f"{false_src}\n\n"
        "def rule(current: str, context: dict) -> str:\n"
        f"    if bool({predicate_expr}):\n"
        f"        return {true_name}(current, context)\n"
        f"    return {false_name}(current, context)\n"
    )


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

    # Optional: reusable partial programs discovered earlier in the same run.
    # These are not answers; they are scored code fragments with residual stats.
    def induction_library_hint(self) -> str:
        return ""

    # Optional: seed candidate programs to evaluate directly before asking the
    # model for new proposals.  These usually come from trusted self-written
    # modules.  Items may be raw code strings or dicts with code/name/description.
    def seed_program_sources(self) -> list[Any]:
        return []

    # Optional: allow generic composition of seed transform programs.
    def compose_seed_programs(self) -> bool:
        return False

    # Optional: allow generic branching over seed transform programs.
    def branch_seed_programs(self) -> bool:
        return False

    # Optional: predicates over observable fields only.  The engine combines
    # them with trusted seeds and validates/refutes the resulting graph.
    def branch_predicate_sources(
        self,
        observations: list[Observation],
    ) -> list[dict[str, str]]:
        return []

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
        self._self_module_registry: SelfModuleRegistry | None = None
        self._self_layer_registry: SelfLayerRegistry | None = None
        self._runtime_layer_sources: dict[str, dict[str, Any]] = {}
        self._residual_kernel = ResidualKernel()
        self.result_log: list[dict[str, Any]] = []

    def _client_lazy(self):
        if self._client is None:
            self._client = make_openai_client()
        return self._client

    # ----- Self-written module memory -----------------------------------
    def _env_enabled(self, name: str, default: str = "1") -> bool:
        value = os.environ.get(name, default).strip().lower()
        return value not in {"0", "false", "no", "off"}

    def _self_modules_enabled(self) -> bool:
        return self._env_enabled("MARS_SELF_WRITE_MODULES", "1")

    def _trusted_self_modules_enabled(self) -> bool:
        """Whether previously promoted hypothesis programs may be loaded.

        Historically ``MARS_SELF_WRITE_MODULES`` controlled both reading and
        writing.  The split flag keeps that behaviour by default while making
        a frozen-language evaluation possible: load the development library,
        but do not mutate it on evaluation tasks.
        """

        if "MARS_LOAD_TRUSTED_MODULES" in os.environ:
            return self._env_enabled("MARS_LOAD_TRUSTED_MODULES", "1")
        return self._self_modules_enabled()

    def _promote_self_modules_enabled(self) -> bool:
        if "MARS_PROMOTE_SELF_MODULES" in os.environ:
            return self._env_enabled("MARS_PROMOTE_SELF_MODULES", "1")
        return self._self_modules_enabled()

    def _candidate_self_modules_enabled(self) -> bool:
        return self._env_enabled("MARS_LOAD_CANDIDATE_MODULES", "0")

    def _quarantine_promotions_enabled(self) -> bool:
        """Persist safe source-task candidates without trusting them yet."""

        return self._env_enabled("MARS_QUARANTINE_PROMOTION", "0")

    def _promotion_probe_enabled(self) -> bool:
        """Record transfer evidence from held-out development tasks only."""

        return self._env_enabled("MARS_PROMOTION_PROBE", "0")

    def _self_layers_enabled(self) -> bool:
        return self._env_enabled("MARS_SELF_WRITE_LAYERS", "1")

    def _trusted_self_layers_enabled(self) -> bool:
        if "MARS_LOAD_TRUSTED_LAYERS" in os.environ:
            return self._env_enabled("MARS_LOAD_TRUSTED_LAYERS", "1")
        return self._self_layers_enabled()

    def _propose_self_layers_enabled(self) -> bool:
        if "MARS_PROPOSE_SELF_LAYERS" in os.environ:
            return self._env_enabled("MARS_PROPOSE_SELF_LAYERS", "1")
        return self._self_layers_enabled()

    def _promote_self_layers_enabled(self) -> bool:
        if "MARS_PROMOTE_SELF_LAYERS" in os.environ:
            return self._env_enabled("MARS_PROMOTE_SELF_LAYERS", "1")
        return self._self_layers_enabled()

    def _candidate_self_layers_enabled(self) -> bool:
        return self._env_enabled("MARS_LOAD_CANDIDATE_LAYERS", "0")

    @staticmethod
    def _bounded_candidate_records(records: list[Any], env_key: str) -> list[Any]:
        """Keep quarantine exploration bounded before candidates are trusted.

        Candidate artifacts are only hypotheses about reusable computation.  A
        fixed, score-ranked budget prevents an early noisy library from
        consuming the complete search budget.  Trusted records are deliberately
        not passed through this function.
        """

        try:
            limit = max(0, int(os.environ.get(env_key, "8")))
        except ValueError:
            limit = 8
        if limit == 0:
            return []
        best_by_hash: dict[str, Any] = {}
        for record in records:
            source_hash = str(getattr(record, "source_hash", ""))
            key = source_hash or str(getattr(record, "name", ""))
            previous = best_by_hash.get(key)
            score = dict(getattr(record, "score", {}) or {})
            quality = (
                float(score.get("loss_mean", 1.0)),
                -float(score.get("exact_rate", score.get("gain", 0.0))),
                float(getattr(record, "created_at", 0.0)),
            )
            if previous is None:
                best_by_hash[key] = record
                continue
            prev_score = dict(getattr(previous, "score", {}) or {})
            prev_quality = (
                float(prev_score.get("loss_mean", 1.0)),
                -float(prev_score.get("exact_rate", prev_score.get("gain", 0.0))),
                float(getattr(previous, "created_at", 0.0)),
            )
            if quality < prev_quality:
                best_by_hash[key] = record
        ranked = sorted(
            best_by_hash.values(),
            key=lambda record: (
                float(dict(getattr(record, "score", {}) or {}).get("loss_mean", 1.0)),
                -float(dict(getattr(record, "score", {}) or {}).get("exact_rate", dict(getattr(record, "score", {}) or {}).get("gain", 0.0))),
                str(getattr(record, "source_hash", "")),
            ),
        )
        return ranked[:limit]

    def _residual_operator_layers_enabled(self) -> bool:
        return self._env_enabled("MARS_RESIDUAL_OPERATOR_LAYERS", "1")

    def _typed_priors_enabled(self) -> bool:
        """Allow a fixed-language ablation without changing an adapter."""

        return self._env_enabled("MARS_TYPED_PRIORS", "1")

    def _residual_kernel_enabled(self) -> bool:
        """Allow residual-kernel features to be isolated from later growth."""

        return self._env_enabled("MARS_RESIDUAL_KERNEL", "1")

    def _residual_class_only_promotion_enabled(self) -> bool:
        """Keep a mechanism experiment free of legacy winner memorization.

        The historical self-module path lifts a strong single-task program into
        a registry.  It remains a useful baseline, but it is not evidence that
        a recurring residual induced a reusable operator.  This switch makes
        those two causal paths independently measurable.
        """

        return self._env_enabled("MARS_RESIDUAL_CLASS_ONLY_PROMOTION", "0")

    def _dpsr_enabled(self) -> bool:
        return self._env_enabled("MARS_DPSR", "1")

    def _nova_enabled(self) -> bool:
        return self._env_enabled("MARS_NOVA", "0")

    def _syndrome_enabled(self) -> bool:
        return self._env_enabled("MARS_SYNDROME", "0")

    def _darwin_enabled(self) -> bool:
        return self._env_enabled("MARS_DARWIN", "0")

    def _ip_genome_enabled(self) -> bool:
        return self._env_enabled("MARS_IP_GENOME", "1")

    def _registry(self) -> SelfModuleRegistry:
        if self._self_module_registry is None:
            self._self_module_registry = SelfModuleRegistry(
                os.environ.get("MARS_SELF_MODULE_ROOT") or None
            )
        return self._self_module_registry

    def _layer_registry(self) -> SelfLayerRegistry:
        if self._self_layer_registry is None:
            self._self_layer_registry = SelfLayerRegistry(
                os.environ.get("MARS_SELF_LAYER_ROOT") or None
            )
        return self._self_layer_registry

    def _layer_namespace(self) -> str:
        return os.environ.get("MARS_SELF_LAYER_NAMESPACE", "universal")

    def _validation_key(
        self,
        adapter: CPIAdapter,
        observations: list[Observation],
    ) -> str:
        explicit = os.environ.get("MARS_VALIDATION_KEY")
        if explicit:
            return explicit[:120]
        payload = {
            "adapter": adapter.name,
            "n": len(observations),
            "evidence": [
                {
                    "inputs": _truncate_repr(obs.inputs, 500),
                    "target": _truncate_repr(obs.target, 500),
                    "context": {
                        k: _truncate_repr(v, 240)
                        for k, v in sorted(obs.context.items())
                        if not k.startswith("__")
                    },
                }
                for obs in observations
            ],
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def _load_self_module_programs(
        self,
        adapter: CPIAdapter,
        train_observations: list[Observation],
    ) -> tuple[list[HypothesisProgram], list[str]]:
        """Load previously self-written modules as ordinary refutable seeds.

        Trusted modules are loaded by default.  Candidate modules can be enabled
        for ablations, but every loaded module is still sandbox-compiled and
        rescored on the current task before it can affect the result.
        """

        if not (
            self._trusted_self_modules_enabled()
            or self._candidate_self_modules_enabled()
        ):
            return [], []
        registry = self._registry()
        records: list[Any] = list(registry.load_trusted(adapter.name))
        if self._candidate_self_modules_enabled():
            records.extend(
                self._bounded_candidate_records(
                    registry.load(adapter.name), "MARS_CANDIDATE_MODULE_BUDGET"
                )
            )
        programs: list[HypothesisProgram] = []
        errors: list[str] = []
        sg = adapter.sandbox_globals()
        seen_hashes: set[str] = set()
        for rec in records:
            source_hash = str(getattr(rec, "source_hash", ""))
            if source_hash and source_hash in seen_hashes:
                continue
            seen_hashes.add(source_hash)
            path = Path(str(getattr(rec, "path", "")))
            try:
                code = path.read_text(encoding="utf-8")
            except Exception as exc:
                errors.append(f"self_module:{getattr(rec, 'name', '?')}: load failed: {exc}")
                continue
            ok, err, fn = _sandbox_compile(code, sg)
            if not ok:
                errors.append(f"self_module:{getattr(rec, 'name', '?')}: {err}")
                continue
            try:
                complexity = 1.0 + min(4.0, len(code) / 1200.0)
            except Exception:
                complexity = 2.0
            tags = ("self_module_trusted",) if hasattr(rec, "mean_loss") else ("self_module_candidate",)
            program = HypothesisProgram(
                name=str(getattr(rec, "name", "self_module")),
                description=str(getattr(rec, "description", "self-written module")),
                code=code,
                fn=fn,
                complexity=complexity,
                tags=tags,
            )
            programs.append(adapter.calibrate(program, train_observations))
        return programs, errors

    def _layer_context(
        self,
        adapter: CPIAdapter,
        observations: list[Observation],
        failure_context: str = "",
    ) -> dict[str, Any]:
        try:
            from mars.skills.interface_profiler import profile_observations

            interface_profile = profile_observations(observations).to_dict()
        except Exception as exc:
            interface_profile = {"error": f"{type(exc).__name__}: {exc}"}
        context_keys: list[str] = []
        for obs in observations:
            for key in obs.context:
                if key.startswith("__"):
                    continue
                if key not in context_keys:
                    context_keys.append(key)
        return {
            "adapter_name": adapter.name,
            "interface_description": adapter.interface_description(),
            "signature_hint": adapter.signature_hint(),
            "n_observations": len(observations),
            "input_type": type(observations[0].inputs).__name__ if observations else "unknown",
            "target_type": type(observations[0].target).__name__ if observations else "unknown",
            "context_keys": context_keys[:24],
            "interface_profile": interface_profile,
            "observations": [
                {
                    "inputs": _truncate_repr(obs.inputs, 500),
                    "target": _truncate_repr(obs.target, 500),
                    "context": {
                        k: _truncate_repr(v, 240)
                        for k, v in obs.context.items()
                        if not k.startswith("__")
                    },
                }
                for obs in observations[:12]
            ],
            # The textual view above is safe for prompts and logs.  Executable
            # layers also need a typed runtime view: serializing a numeric
            # mapping into a string prevents a generic measurement operator
            # from ever reading its keys or values.  This view exists only in
            # the sandbox call; it is never embedded in a persisted layer.
            "observations_typed": [
                {
                    "inputs": obs.inputs,
                    "target": obs.target,
                    "context": {
                        key: value
                        for key, value in obs.context.items()
                        if not key.startswith("__")
                    },
                }
                for obs in observations[:12]
            ],
            "inference_prior_genome": self._ip_genome_context(adapter, observations),
            "failure_context": failure_context[:2500],
        }

    def _load_regulatory_genome_for_ip(self):
        try:
            from mars.darwin.regulatory import RegulatoryGenome, default_regulatory_genome
        except Exception:
            return None
        genome_path = os.environ.get("MARS_REGULATORY_GENOME", "").strip()
        if genome_path:
            try:
                return RegulatoryGenome.load_json(genome_path)
            except Exception:
                return default_regulatory_genome()
        return default_regulatory_genome()

    def _ip_genome_context(
        self,
        adapter: CPIAdapter,
        observations: list[Observation],
    ) -> dict[str, Any]:
        """Develop the IP-Genome into prompt/code/contract guidance.

        This is API-only learning glue: the frozen model is not updated, but the
        learned regulatory genome changes the inference prior it sees.
        """

        if not self._ip_genome_enabled() or not observations:
            return {}
        try:
            from mars.darwin import DarwinSynthesizer
        except Exception:
            return {}
        genome = self._load_regulatory_genome_for_ip()
        if genome is None:
            return {}
        try:
            result = DarwinSynthesizer(
                max_programs=0,
                regulatory_genome=genome,
            ).synthesize(
                signature_hint=adapter.signature_hint(),
                observations=observations,
                interface_name=adapter.name,
                question=str(getattr(adapter, "question", "")),
                column_descriptions=dict(getattr(adapter, "column_descriptions", {}) or {}),
                domain_context=str(getattr(adapter, "domain_knowledge", "")),
            )
        except Exception:
            return {}
        for row in result.trace:
            dev = row.get("regulatory_development") if isinstance(row, dict) else None
            if not isinstance(dev, dict):
                continue
            active = [
                str(item.get("gene"))
                for item in dev.get("active_genes", []) or []
                if isinstance(item, dict) and item.get("gene")
            ]
            return {
                "genome": dev.get("regulatory_genome"),
                "generation": dev.get("generation"),
                "lineage": dev.get("lineage", []),
                "active_genes": active,
                "prompt_scaffolds": list(dev.get("emitted_prompt_scaffolds", []) or []),
                "code_probes": list(dev.get("emitted_code_probes", []) or []),
                "contract_validators": list(dev.get("emitted_contract_validators", []) or []),
                "hypothesis_families": list(dev.get("emitted_families", []) or []),
                "suppressed_families": list(dev.get("suppressed_families", []) or []),
            }
        return {}

    def _format_ip_genome_context(
        self,
        adapter: CPIAdapter,
        observations: list[Observation],
    ) -> str:
        context = self._ip_genome_context(adapter, observations)
        if not context:
            return ""
        return (
            "\nINFERENCE-PRIOR GENOME EXPRESSION:\n"
            "The following prompt scaffolds, executable probe types, contract "
            "validators, and hypothesis families were developed from the task "
            "interface and typed failure algebra. Use them as search bias, not "
            "as benchmark-specific answers.\n"
            f"{json.dumps(context, indent=2, ensure_ascii=False, default=str)[:2500]}\n"
        )

    def _program_source_dict(self, program: HypothesisProgram) -> dict[str, Any]:
        return {
            "name": program.name,
            "description": program.description,
            "complexity": program.complexity,
            "code": program.code,
            "tags": list(program.tags),
        }

    def _programs_from_layer_sources(
        self,
        adapter: CPIAdapter,
        sources: list[Any],
        train_observations: list[Observation],
        *,
        layer_name: str,
        layer_origin: str = "trusted",
    ) -> tuple[list[HypothesisProgram], list[str]]:
        programs: list[HypothesisProgram] = []
        errors: list[str] = []
        sg = adapter.sandbox_globals()
        for i, item in enumerate(sources[:64]):
            if isinstance(item, str):
                code = item.strip()
                name = f"{layer_name}_program_{i}"
                description = f"program emitted by self layer {layer_name}"
                complexity = 2.0
            elif isinstance(item, dict):
                code = str(item.get("code", "")).strip()
                name = str(item.get("name", f"{layer_name}_program_{i}"))
                description = str(
                    item.get("description", f"program emitted by self layer {layer_name}")
                )
                try:
                    complexity = float(item.get("complexity", 2.0))
                except Exception:
                    complexity = 2.0
            else:
                continue
            if not code:
                continue
            ok, err, fn = _sandbox_compile(code, sg)
            if not ok:
                errors.append(f"self_layer:{layer_name}:{name}: {err}")
                continue
            programs.append(
                adapter.calibrate(
                    HypothesisProgram(
                        name=name,
                        description=description,
                        code=code,
                        fn=fn,
                        complexity=complexity + 0.4,
                        tags=("self_layer_program", layer_name, layer_origin),
                    ),
                    train_observations,
                )
            )
        return programs, errors

    def _programs_from_layer_operators(
        self,
        adapter: CPIAdapter,
        layer_outputs: list[dict[str, Any]],
        base_programs: list[HypothesisProgram],
        train_observations: list[Observation],
        *,
        failure_context: str = "",
        max_operator_outputs: int = 96,
    ) -> tuple[list[HypothesisProgram], list[str]]:
        """Run self-written search operators over current candidate programs."""

        if not layer_outputs or not base_programs:
            return [], []
        programs: list[HypothesisProgram] = []
        errors: list[str] = []
        context = self._layer_context(adapter, train_observations, failure_context)
        source_pool = [self._program_source_dict(p) for p in base_programs[:96]]
        for output in layer_outputs[:12]:
            layer_name = str(output.get("_layer_name") or "self_layer")
            layer_origin = str(output.get("_layer_origin") or "trusted")
            operator_sources = list(output.get("operator_sources") or [])
            for i, item in enumerate(operator_sources[:16]):
                if isinstance(item, str):
                    code = item.strip()
                    op_name = f"{layer_name}_operator_{i}"
                    op_description = f"operator emitted by self layer {layer_name}"
                elif isinstance(item, dict):
                    code = str(item.get("code", "")).strip()
                    op_name = str(item.get("name", f"{layer_name}_operator_{i}"))
                    op_description = str(
                        item.get("description", f"operator emitted by self layer {layer_name}")
                    )
                else:
                    continue
                if not code:
                    continue
                ok, err, transform = _sandbox_compile_named(code, "transform")
                if not ok:
                    errors.append(f"self_layer_operator:{layer_name}:{op_name}: {err}")
                    continue
                try:
                    new_sources = transform(list(source_pool), dict(context))
                except Exception as exc:
                    errors.append(f"self_layer_operator:{layer_name}:{op_name}: run failed: {exc}")
                    continue
                if not isinstance(new_sources, list):
                    errors.append(f"self_layer_operator:{layer_name}:{op_name}: output is not list")
                    continue
                generated, gen_errors = self._programs_from_layer_sources(
                    adapter,
                    new_sources[:max_operator_outputs],
                    train_observations,
                    layer_name=f"{layer_name}_{op_name}",
                    layer_origin=f"{layer_origin}_operator",
                )
                for program in generated:
                    program.description = f"{program.description} (via operator: {op_description})"
                    program.tags = (
                        "self_layer_program",
                        layer_name,
                        layer_origin,
                        "operator",
                        op_name,
                    )
                programs.extend(generated)
                errors.extend(gen_errors)
        return programs, errors

    def _execute_layer_source(
        self,
        adapter: CPIAdapter,
        train_observations: list[Observation],
        *,
        layer_name: str,
        code: str,
        layer_origin: str,
        failure_context: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> tuple[list[HypothesisProgram], list[str], dict[str, Any] | None]:
        ok_static, reason = validate_self_layer_source(code)
        if not ok_static:
            return [], [f"self_layer:{layer_name}: {reason}"], None
        ok, err, build_layer = _sandbox_compile_named(code, "build_layer")
        if not ok:
            return [], [f"self_layer:{layer_name}: {err}"], None
        context = self._layer_context(adapter, train_observations, failure_context)
        try:
            output = build_layer(dict(context))
        except Exception as exc:
            return [], [f"self_layer:{layer_name}: run failed: {exc}"], None
        if not isinstance(output, dict):
            return [], [f"self_layer:{layer_name}: output is not dict"], None
        output["_layer_name"] = layer_name
        output["_layer_origin"] = layer_origin
        if layer_origin in {"proposed", "candidate"}:
            self._runtime_layer_sources[layer_name] = {
                "code": code,
                "metadata": dict(metadata or {}),
                "output": output,
            }
        programs, errors = self._programs_from_layer_sources(
            adapter,
            list(output.get("program_sources") or []),
            train_observations,
            layer_name=layer_name,
            layer_origin=layer_origin,
        )
        # Parsing an emitted program is not enough: weak models often write a
        # function that closes over ``context`` or ``observations`` although
        # neither belongs to its declared signature.  Execute each source on
        # the available interface values before it can enter quarantine.
        runnable: list[HypothesisProgram] = []
        for program in programs:
            try:
                ran = False
                for observation in train_observations[:3]:
                    adapter.execute(program, observation)
                    ran = True
            except Exception as exc:
                errors.append(
                    f"self_layer:{layer_name}:{program.name}: runtime failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            if ran:
                runnable.append(program)
        return runnable, errors, output

    def _load_self_layer_programs(
        self,
        adapter: CPIAdapter,
        train_observations: list[Observation],
        failure_context: str = "",
    ) -> tuple[list[HypothesisProgram], list[str], list[dict[str, Any]]]:
        """Run trusted self-written layers and compile their emitted programs."""

        if not (
            self._trusted_self_layers_enabled()
            or self._candidate_self_layers_enabled()
        ):
            return [], [], []
        registry = self._layer_registry()
        records: list[Any] = list(registry.load_trusted(self._layer_namespace()))
        if self._candidate_self_layers_enabled():
            records.extend(
                self._bounded_candidate_records(
                    registry.load(self._layer_namespace()), "MARS_CANDIDATE_LAYER_BUDGET"
                )
            )
        programs: list[HypothesisProgram] = []
        errors: list[str] = []
        outputs: list[dict[str, Any]] = []
        seen_hashes: set[str] = set()
        for rec in records:
            source_hash = str(getattr(rec, "source_hash", ""))
            if source_hash and source_hash in seen_hashes:
                continue
            seen_hashes.add(source_hash)
            path = Path(str(getattr(rec, "path", "")))
            try:
                code = path.read_text(encoding="utf-8")
            except Exception as exc:
                errors.append(f"self_layer:{getattr(rec, 'name', '?')}: load failed: {exc}")
                continue
            layer_programs, layer_errors, output = self._execute_layer_source(
                adapter,
                train_observations,
                layer_name=str(getattr(rec, "name", "self_layer")),
                code=code,
                layer_origin="trusted" if hasattr(rec, "mean_loss") else "candidate",
                failure_context=failure_context,
                metadata={"record": str(getattr(rec, "name", "self_layer"))},
            )
            # A loaded operator is a structural hypothesis, not a fully fitted
            # answer.  Apply the adapter's local data calibration exactly as we
            # do for newly proposed executable programs; otherwise an induced
            # law cannot recover task-local constants at all.
            layer_programs = [adapter.calibrate(program, train_observations)
                              for program in layer_programs]
            programs.extend(layer_programs)
            errors.extend(layer_errors)
            if output is not None:
                outputs.append(output)
        return programs, errors, outputs

    def _build_self_layer_prompt(
        self,
        adapter: CPIAdapter,
        observations: list[Observation],
        failure_context: str = "",
    ) -> str:
        context = self._layer_context(adapter, observations, failure_context)
        ip_context = self._format_ip_genome_context(adapter, observations)
        n_layers = int(os.environ.get("MARS_SELF_LAYER_PROPOSALS", "2"))
        return f"""You are MARS's architecture self-writer.

Your job is NOT to solve one benchmark directly. Write reusable architecture
layers that transform a generic task context into candidate executable
hypotheses, data-derived signals, or residual hints.

LAYER CONTRACT:
def build_layer(context: dict) -> dict:
    ...

The function must return a dict with optional keys:
- "program_sources": list of dicts with name, description, complexity, code
- "operator_sources": list of dicts with name, description, code. Each code
  defines def transform(program_sources: list, context: dict) -> list
- "signals": list[str]
- "residual_hints": list[str]
- "metadata": dict

Every emitted program must match this exact function signature:
{adapter.signature_hint()}

The layer receives only this generic context:
{json.dumps(context, indent=2, ensure_ascii=False, default=str)[:5000]}
{ip_context}

Safety rules:
- no file/network/process access
- no eval/exec/open/import except math/statistics/itertools
- no benchmark-specific hidden answers
- use only context fields and observations
- emitted programs will be sandboxed and scored on held-out observations
- emitted operators will receive only existing program source dictionaries and
  the same generic context, then their generated programs will be sandboxed and
  scored on held-out observations

Propose up to {n_layers} diverse self-layers. Prefer layers that generate a
family of small checkable programs, not one brittle answer.

Return ONLY JSON:
{{
  "layers": [
    {{
      "name": "snake_case_layer_name",
      "layer_type": "program_source_generator",
      "description": "what reusable search transformation this layer adds",
      "contract": {{"input": "context: dict", "output": "dict(program_sources, signals, residual_hints, metadata)"}},
      "code": "def build_layer(context: dict) -> dict:\\n    ..."
    }}
  ]
}}"""

    def _propose_self_layers(
        self,
        adapter: CPIAdapter,
        observations: list[Observation],
        failure_context: str = "",
    ) -> tuple[list[HypothesisProgram], list[dict], list[str], list[dict[str, Any]]]:
        if not self._propose_self_layers_enabled() or self.n_proposals <= 0:
            return [], [], [], []
        n_layers = int(os.environ.get("MARS_SELF_LAYER_PROPOSALS", "2"))
        if n_layers <= 0:
            return [], [], [], []
        prompt = self._build_self_layer_prompt(adapter, observations, failure_context)
        raw = call_llm(
            self._client_lazy(),
            model=self.model,
            system=(
                "You write safe, reusable Python architecture layers for MARS. "
                "Return only valid JSON."
            ),
            user=prompt,
            max_tokens=4500,
            temperature=min(1.0, max(0.2, self.temperature)),
        )
        proposals = _parse_layers(raw)[:n_layers]
        programs: list[HypothesisProgram] = []
        errors: list[str] = []
        outputs: list[dict[str, Any]] = []
        for i, item in enumerate(proposals):
            if not isinstance(item, dict):
                continue
            code = str(item.get("code", "")).strip()
            name = _safe_name(str(item.get("name", f"generated_layer_{i}")))
            if not code:
                errors.append(f"self_layer:{name}: no code")
                continue
            layer_programs, layer_errors, output = self._execute_layer_source(
                adapter,
                observations,
                layer_name=name,
                code=code,
                layer_origin="proposed",
                failure_context=failure_context,
                metadata=item,
            )
            programs.extend(layer_programs)
            errors.extend(layer_errors)
            if output is not None:
                outputs.append(output)
        return programs, proposals, errors, outputs

    def _layer_source_from_program(
        self,
        adapter: CPIAdapter,
        program: HypothesisProgram,
        score: ProgramScore,
    ) -> str:
        signature = adapter.signature_hint()
        program_source = {
            "name": program.name,
            "description": program.description,
            "complexity": min(5.0, float(program.complexity) + 0.2),
            "code": program.code,
        }
        signal = (
            f"Self-written reusable layer for signature {signature}; "
            f"source winner={program.name}, loss={score.loss_mean:.4f}, "
            f"exact={score.exact_rate:.3f}."
        )
        return (
            "def build_layer(context: dict) -> dict:\n"
            f"    if str(context.get('signature_hint', '')) != {signature!r}:\n"
            "        return {'program_sources': [], 'signals': [], 'residual_hints': []}\n"
            "    return {\n"
            f"        'program_sources': [{program_source!r}],\n"
            f"        'signals': [{signal!r}],\n"
            "        'residual_hints': [],\n"
            "        'metadata': {'origin': 'winner_to_layer', 'scope': 'signature'},\n"
            "    }\n"
        )

    def _promote_self_layers(
        self,
        adapter: CPIAdapter,
        winners: list[tuple[HypothesisProgram, ProgramScore]],
        observations: list[Observation],
        baseline_loss: float | None = None,
    ) -> list[str]:
        """Promote winning programs into reusable layer generators."""

        if not self._promote_self_layers_enabled() or not winners:
            return []
        registry = self._layer_registry()
        validation_key = self._validation_key(adapter, observations)
        notes: list[str] = []
        if self._promotion_probe_enabled():
            winners = [
                (program, score)
                for program, score in winners
                if "self_layer_program" in program.tags
                and len(program.tags) > 2
                and str(program.tags[2]).startswith("candidate")
            ]
        for program, score in winners[:2]:
            if "self_layer_program" in program.tags:
                layer_name = program.tags[1] if len(program.tags) > 1 else ""
                source_record = self._runtime_layer_sources.get(layer_name)
                if not source_record:
                    continue
                code = str(source_record.get("code", ""))
                metadata = dict(source_record.get("metadata") or {})
                name = str(metadata.get("name") or layer_name or f"{adapter.name}_generated_layer")
                layer_type = str(metadata.get("layer_type") or "program_source_generator")
                contract = dict(metadata.get("contract") or {
                    "input": "context: dict",
                    "output": "dict(program_sources, signals, residual_hints, metadata)",
                    "scope": "generated_layer",
                    "adapter": adapter.name,
                })
                description = str(
                    metadata.get("description")
                    or f"Generated layer whose emitted program {program.name} won after refutation."
                )
            else:
                code = self._layer_source_from_program(adapter, program, score)
                name = f"{adapter.name}_{program.name}_layer"
                layer_type = "program_source_generator"
                contract = {
                    "input": "context: dict",
                    "output": "dict(program_sources, signals, residual_hints, metadata)",
                    "scope": "signature_hint",
                    "adapter": adapter.name,
                }
                description = (
                    f"Emit reusable candidate program {program.name} for matching "
                    f"interface signature; always rescored before use."
                )
            payload = {
                "gain": max(0.0, 1.0 - float(score.loss_mean)),
                "loss_mean": score.loss_mean,
                "exact_rate": score.exact_rate,
                "validation_key": validation_key,
                "source_program": program.name,
                "relative_gain": (
                    float(baseline_loss) - float(score.loss_mean)
                    if baseline_loss is not None
                    else None
                ),
                "evidence_phase": (
                    "promotion_probe" if self._promotion_probe_enabled() else "induction"
                ),
            }
            candidate_thresholds = (
                self._quarantine_promotions_enabled() or self._promotion_probe_enabled()
            )
            record = registry.promote(
                namespace=self._layer_namespace(),
                name=name,
                layer_type=layer_type,
                code=code,
                contract=contract,
                description=description,
                score=payload,
                min_gain=float(os.environ.get(
                    "MARS_CANDIDATE_LAYER_MIN_GAIN" if candidate_thresholds else "MARS_SELF_LAYER_MIN_GAIN",
                    "0" if candidate_thresholds else "0.25",
                )),
                max_loss_mean=float(os.environ.get(
                    "MARS_CANDIDATE_LAYER_MAX_LOSS" if candidate_thresholds else "MARS_SELF_LAYER_MAX_LOSS",
                    "1" if candidate_thresholds else "0.05",
                )),
            )
            if record is None:
                continue
            notes.append(f"promoted self layer {record.name} for {adapter.name}")
        if notes:
            trusted = registry.refresh_trusted_manifest(
                self._layer_namespace(),
                min_validation_keys=int(os.environ.get("MARS_SELF_LAYER_MIN_KEYS", "2")),
                min_mean_gain=float(os.environ.get("MARS_SELF_LAYER_TRUSTED_MIN_GAIN", "0.25")),
                max_mean_loss=float(os.environ.get("MARS_SELF_LAYER_TRUSTED_MAX_LOSS", "0.05")),
                min_probe_validation_keys=int(
                    os.environ.get("MARS_SELF_LAYER_MIN_PROBE_KEYS", "0")
                ),
                min_mean_probe_relative_gain=(
                    float(os.environ["MARS_SELF_LAYER_MIN_PROBE_RELATIVE_GAIN"])
                    if "MARS_SELF_LAYER_MIN_PROBE_RELATIVE_GAIN" in os.environ
                    else None
                ),
            )
            if trusted:
                notes.append(
                    f"trusted self-layer library size for {self._layer_namespace()}: {len(trusted)}"
                )
        return notes

    def _promote_self_modules(
        self,
        adapter: CPIAdapter,
        winners: list[tuple[HypothesisProgram, ProgramScore]],
        observations: list[Observation],
        baseline_loss: float | None = None,
    ) -> list[str]:
        """Persist high-scoring hypotheses as future reusable modules."""

        if not self._promote_self_modules_enabled() or not winners:
            return []
        registry = self._registry()
        validation_key = self._validation_key(adapter, observations)
        notes: list[str] = []
        if self._promotion_probe_enabled():
            winners = [
                (program, score)
                for program, score in winners
                if "self_module_candidate" in program.tags
            ]
        for program, score in winners[:3]:
            if "self_module_trusted" in program.tags:
                continue
            payload = {
                "loss_mean": score.loss_mean,
                "exact_rate": score.exact_rate,
                "mdl_score": score.mdl_score,
                "complexity": score.complexity,
                "n_scored": score.n_scored,
                "validation_key": validation_key,
                "relative_gain": (
                    float(baseline_loss) - float(score.loss_mean)
                    if baseline_loss is not None
                    else None
                ),
                "evidence_phase": (
                    "promotion_probe" if self._promotion_probe_enabled() else "induction"
                ),
            }
            candidate_thresholds = (
                self._quarantine_promotions_enabled() or self._promotion_probe_enabled()
            )
            record = registry.promote(
                namespace=adapter.name,
                name=program.name,
                code=program.code,
                description=program.description,
                score=payload,
                min_exact_rate=float(os.environ.get(
                    "MARS_CANDIDATE_MODULE_MIN_EXACT" if candidate_thresholds else "MARS_SELF_MODULE_MIN_EXACT",
                    "0" if candidate_thresholds else "0.75",
                )),
                max_loss_mean=float(os.environ.get(
                    "MARS_CANDIDATE_MODULE_MAX_LOSS" if candidate_thresholds else "MARS_SELF_MODULE_MAX_LOSS",
                    "1" if candidate_thresholds else "0.05",
                )),
            )
            if record is None:
                continue
            notes.append(f"promoted self module {record.name} for {adapter.name}")
        if notes:
            trusted = registry.refresh_trusted_manifest(
                adapter.name,
                min_validation_keys=int(os.environ.get("MARS_SELF_MODULE_MIN_KEYS", "2")),
                min_mean_exact_rate=float(os.environ.get("MARS_SELF_MODULE_TRUSTED_MIN_EXACT", "0.75")),
                max_mean_loss=float(os.environ.get("MARS_SELF_MODULE_TRUSTED_MAX_LOSS", "0.05")),
                min_probe_validation_keys=int(
                    os.environ.get("MARS_SELF_MODULE_MIN_PROBE_KEYS", "0")
                ),
                min_mean_probe_relative_gain=(
                    float(os.environ["MARS_SELF_MODULE_MIN_PROBE_RELATIVE_GAIN"])
                    if "MARS_SELF_MODULE_MIN_PROBE_RELATIVE_GAIN" in os.environ
                    else None
                ),
            )
            if trusted:
                notes.append(f"trusted library size for {adapter.name}: {len(trusted)}")
        return notes

    # ----- Cross-task residual-class induction ---------------------------
    def _residual_class_ledger(self) -> ResidualClassLedger:
        raw_root = os.environ.get("MARS_RESIDUAL_LEDGER_ROOT")
        root = Path(raw_root) if raw_root else self._layer_registry().root / "residual_classes"
        return ResidualClassLedger(root)

    def _residual_fingerprint(
        self,
        adapter: CPIAdapter,
        observations: list[Observation],
        best: tuple[HypothesisProgram, ProgramScore] | None,
    ) -> dict[str, Any]:
        """Describe failure geometry without task text, values, or field names."""

        context_shapes = sorted({
            tuple(sorted(type(value).__name__ for value in observation.context.values()))
            for observation in observations[:8]
        })
        loss = 1.0 if best is None else float(best[1].loss_mean)
        loss_band = "high" if loss >= 0.66 else "mixed" if loss >= 0.20 else "low"
        geometry = self._numeric_residual_geometry(adapter, observations, best[0] if best else None)
        # Signature comments often enumerate task-local field names.  Retaining
        # the resulting list length fragments one typed interface into a
        # separate residual class for every arity.  Class identity needs the
        # callable contract, while concrete fields remain runtime data.
        signature_contract = adapter.signature_hint().split("#", 1)[0].strip()
        return {
            "input_type": type(observations[0].inputs).__name__ if observations else "unknown",
            "target_type": type(observations[0].target).__name__ if observations else "unknown",
            "signature": re.sub(
                r"\b[A-Za-z_][A-Za-z0-9_]*\b", "slot", signature_contract
            ),
            "context_value_shapes": [list(shape) for shape in context_shapes[:3]],
            "loss_band": loss_band,
            "support_bucket": "small" if len(observations) < 8 else "medium" if len(observations) < 64 else "large",
            "residual_geometry": geometry,
        }

    @staticmethod
    def _numeric_residual_geometry(
        adapter: CPIAdapter,
        observations: list[Observation],
        best_program: HypothesisProgram | None,
    ) -> str:
        """Classify an interface-level numeric residual without retaining data.

        The classifier records only a coarse, permutation-invariant category.
        It deliberately avoids variable names, task text, fitted coefficients,
        and individual examples.  A monomial chart is useful when the observed
        response has stable log-linear structure; an additive chart is useful
        when it does not.
        """

        numeric_rows: list[tuple[dict[str, float], float]] = []
        for observation in observations:
            if not isinstance(observation.inputs, dict):
                return "non_numeric_or_non_dict"
            try:
                inputs = {str(key): float(value) for key, value in observation.inputs.items()}
                target = float(observation.target)
            except (TypeError, ValueError):
                return "non_numeric_or_non_dict"
            if not inputs or not math.isfinite(target):
                return "non_numeric_or_non_dict"
            numeric_rows.append((inputs, target))
        if len(numeric_rows) < 4:
            return "insufficient_numeric_support"

        positives = (
            all(target > 0 for _, target in numeric_rows)
            and all(value > 0 for inputs, _ in numeric_rows for value in inputs.values())
        )
        if not positives:
            return "additive_or_signed_numeric"

        common_keys = sorted(set.intersection(*(set(inputs) for inputs, _ in numeric_rows)))
        log_targets = [math.log(target) for _, target in numeric_rows]

        def correlation(left: list[float], right: list[float]) -> float:
            if len(left) != len(right) or len(left) < 2:
                return 0.0
            mean_left = sum(left) / len(left)
            mean_right = sum(right) / len(right)
            numerator = sum((x - mean_left) * (y - mean_right) for x, y in zip(left, right))
            denom_left = sum((x - mean_left) ** 2 for x in left)
            denom_right = sum((y - mean_right) ** 2 for y in right)
            if denom_left <= 1e-12 or denom_right <= 1e-12:
                return 0.0
            return numerator / math.sqrt(denom_left * denom_right)

        max_log_correlation = max(
            (abs(correlation([math.log(inputs[key]) for inputs, _ in numeric_rows], log_targets))
             for key in common_keys),
            default=0.0,
        )
        if best_program is not None:
            log_ratio_residuals: list[float] = []
            for observation, (_, target) in zip(observations, numeric_rows):
                try:
                    prediction = float(adapter.execute(best_program, observation))
                except Exception:
                    continue
                if prediction > 0 and math.isfinite(prediction):
                    log_ratio_residuals.append(math.log(target / prediction))
            if len(log_ratio_residuals) >= 4:
                spread = max(log_ratio_residuals) - min(log_ratio_residuals)
                if spread <= 0.15:
                    return "multiplicative_scale_only"
        return (
            "positive_log_structured"
            if max_log_correlation >= 0.55
            else "positive_nonlog_structured"
        )

    def _induce_residual_class_layer(
        self,
        adapter: CPIAdapter,
        observations: list[Observation],
        ranked: list[tuple[HypothesisProgram, ProgramScore]],
    ) -> list[str]:
        """Propose one parameter-free operator after a residual class recurs.

        The proposal context intentionally excludes task identifiers, natural
        language questions, raw observations, and answers.  Consequently the
        induced source must operate through the generic runtime ``context``.
        It is still only a quarantine candidate until independent probes show
        it improves over the base language.
        """

        if (
            not self._quarantine_promotions_enabled()
            or self._promotion_probe_enabled()
            or not self._env_enabled("MARS_RESIDUAL_CLASS_INDUCTION", "0")
        ):
            return []
        fingerprint = self._residual_fingerprint(
            adapter, observations, ranked[0] if ranked else None
        )
        ledger = self._residual_class_ledger()
        validation_key = self._validation_key(adapter, observations)
        class_id, support = ledger.observe(
            validation_key=validation_key, fingerprint=fingerprint
        )
        try:
            min_support = max(2, int(os.environ.get("MARS_RESIDUAL_CLASS_MIN_SUPPORT", "2")))
        except ValueError:
            min_support = 2
        if support < min_support or not ledger.claim(class_id):
            return []
        prior = ledger.latest_outcome(class_id) or {}
        prior_feedback = str(prior.get("detail", "none"))[:400]
        families = admissible_families(
            input_type=str(fingerprint["input_type"]),
            target_type=str(fingerprint["target_type"]),
            signature_hint=adapter.signature_hint(),
        )
        if not families:
            ledger.record_outcome(class_id, "no_typed_family", "no compatible typed operator family")
            return [f"residual class {class_id}: no compatible typed family"]
        prompt = f"""Choose one reusable typed operator plan for a recurring
scientific residual class. You select a plan; a trusted compiler, not you,
will turn it into code and execute it on unseen tasks.

Residual class: {json.dumps(fingerprint, sort_keys=True)}
Admissible operator families: {json.dumps(families)}
Prior compiler feedback: {prior_feedback}

For numeric dict-to-scalar interfaces, choose positive_monomial_lattice only
for residual geometry `positive_log_structured` or `multiplicative_scale_only`;
choose positive_additive_lattice for `positive_nonlog_structured` or
`additive_or_signed_numeric`. Choose exactly one family from the admissible list. Do not name a dataset,
field, task, source value, or answer. Return only:
{{"plans": [{{"family": "one admissible family", "description": "short generic role"}}]}}"""
        raw = call_llm(
            self._client_lazy(),
            model=self.model,
            system="You select only a typed, domain-agnostic executable operator family. Return valid JSON.",
            user=prompt,
            max_tokens=300,
            temperature=min(0.35, max(0.1, self.temperature)),
        )
        plans = _parse_operator_plans(raw)
        item = compile_typed_operator_plan(
            plans[0] if plans else {},
            input_type=str(fingerprint["input_type"]),
            target_type=str(fingerprint["target_type"]),
            signature_hint=adapter.signature_hint(),
        )
        if item is None:
            ledger.record_outcome(class_id, "invalid_typed_plan", str(raw))
            return [f"residual class {class_id}: invalid typed operator plan"]
        code = str(item.get("code", "")).strip()
        safe, reason = validate_self_layer_source(code)
        if not safe:
            ledger.record_outcome(class_id, "static_reject", reason)
            return [f"residual class {class_id}: rejected operator ({reason})"]
        emitted, emitted_errors, output = self._execute_layer_source(
            adapter,
            observations,
            layer_name=f"residual_class_{class_id}",
            code=code,
            layer_origin="proposed",
            metadata=item,
        )
        if output is None or not emitted:
            detail = "; ".join(emitted_errors[:3]) or "no executable hypotheses emitted"
            ledger.record_outcome(class_id, "semantic_reject", detail)
            return [f"residual class {class_id}: rejected non-executable operator"]
        name = _safe_name(str(item.get("name", f"residual_class_{class_id}")))
        record = self._layer_registry().promote(
            namespace=self._layer_namespace(),
            name=f"residual_class_{name}",
            layer_type="residual_class_operator",
            code=code,
            contract={
                "input": "context: dict",
                "output": "dict(program_sources, operator_sources, signals, residual_hints, metadata)",
                "scope": "typed_residual_class",
                "class_id": class_id,
            },
            description=str(item.get("description", "operator induced from a recurring typed residual class")),
            score={
                "gain": 0.0,
                "loss_mean": 1.0,
                "validation_key": validation_key,
                "evidence_phase": "induction",
                "residual_class_id": class_id,
                "residual_class_support": support,
            },
            min_gain=0.0,
            max_loss_mean=1.0,
        )
        if record is None:
            ledger.record_outcome(class_id, "persistence_reject")
            return [f"residual class {class_id}: operator persistence rejected"]
        ledger.record_outcome(class_id, "quarantined", record.name)
        return [
            f"quarantined residual-class operator {record.name} "
            f"(class={class_id}, support={support})"
        ]

    def _finalize_result(
        self,
        *,
        adapter: CPIAdapter,
        observations: list[Observation],
        programs: list[HypothesisProgram],
        proposals_raw: list[dict],
        errors: list[str],
        t0: float,
        rounds: int,
        n_proposed: int | None = None,
    ) -> CPIResult:
        ranked_full = self._rank(programs, observations, adapter) if programs else []
        winners = ranked_full[:5]
        promotion_pool = winners
        baseline_loss: float | None = None
        if self._promotion_probe_enabled():
            def is_candidate(program: HypothesisProgram) -> bool:
                tags = set(program.tags)
                return (
                    "self_module_candidate" in tags
                    or (
                        "self_layer_program" in tags
                        and any(str(tag).startswith("candidate") for tag in program.tags)
                    )
                )

            base_scores = [score for program, score in ranked_full if not is_candidate(program)]
            baseline_loss = base_scores[0].loss_mean if base_scores else None
            try:
                evidence_budget = max(
                    1, int(os.environ.get("MARS_PROMOTION_EVIDENCE_CANDIDATES", "8"))
                )
            except ValueError:
                evidence_budget = 8
            promotion_pool = [
                (program, score)
                for program, score in ranked_full
                if is_candidate(program)
            ][:evidence_budget]
        if (
            self._residual_class_only_promotion_enabled()
            and not self._promotion_probe_enabled()
        ):
            layer_notes = []
            promotion_notes = []
        else:
            layer_notes = self._promote_self_layers(
                adapter, promotion_pool, observations, baseline_loss=baseline_loss
            )
            promotion_notes = self._promote_self_modules(
                adapter, promotion_pool, observations, baseline_loss=baseline_loss
            )
        residual_class_notes = self._induce_residual_class_layer(
            adapter, observations, ranked_full
        )
        report = adapter.render_report(winners, observations) if winners else ""
        all_errors = errors + layer_notes + promotion_notes + residual_class_notes
        wall_time_s = time.time() - t0
        try:
            from mars.skills.supermetrics import compute_cpi_supermetrics

            supermetrics = compute_cpi_supermetrics(
                benchmark=adapter.name,
                n_observations=len(observations),
                n_proposed=len(proposals_raw) if n_proposed is None else n_proposed,
                n_valid=len(programs),
                winners=winners,
                report=report,
                errors=all_errors,
                wall_time_s=wall_time_s,
                rounds=rounds,
            ).to_dict()
        except Exception as exc:
            supermetrics = {"error": f"{type(exc).__name__}: {exc}"}
        try:
            from mars.induction.epistemic_compiler import certify_epistemic_result

            certificate = certify_epistemic_result(
                adapter_name=adapter.name,
                signature_hint=adapter.signature_hint(),
                interface_description=adapter.interface_description(),
                observations=observations,
                winners=winners,
                programs=programs,
                proposals_raw=proposals_raw,
            )
            supermetrics["epistemic_certificate"] = certificate.to_dict()
        except Exception as exc:
            supermetrics["epistemic_certificate_error"] = f"{type(exc).__name__}: {exc}"
        result = CPIResult(
            benchmark=adapter.name,
            n_observations=len(observations),
            n_proposed=len(proposals_raw) if n_proposed is None else n_proposed,
            n_valid=len(programs),
            winners=winners,
            report=report,
            proposals_raw=proposals_raw,
            errors=all_errors,
            wall_time_s=wall_time_s,
            rounds=rounds,
            supermetrics=supermetrics,
        )
        try:
            self.result_log.append(result.to_dict())
        except Exception:
            pass
        return result

    def _with_darwin_trace_proposals(
        self,
        adapter: CPIAdapter,
        train_observations: list[Observation],
        proposals_raw: list[dict],
    ) -> list[dict]:
        if any(isinstance(p, dict) and p.get("kind") == "darwin_trace" for p in proposals_raw):
            return proposals_raw
        trace = self._darwin_trace_only(adapter, train_observations)
        if not trace:
            return proposals_raw
        return list(proposals_raw) + trace

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
        try:
            from mars.skills.interface_profiler import profile_prompt

            interface_profile = profile_prompt(observations)
        except Exception as exc:
            interface_profile = f"\nUNIVERSAL INTERFACE PROFILE unavailable: {type(exc).__name__}: {exc}"
        library_hint = adapter.induction_library_hint()
        ip_context = self._format_ip_genome_context(adapter, observations)
        prompt = f"""{adapter.interface_description()}

FUNCTION SIGNATURE (your functions MUST match exactly):
{adapter.signature_hint()}

OBSERVED EVIDENCE (inputs + context → target):
{json.dumps(examples, indent=2, ensure_ascii=False, default=str)[:3500]}
{interface_profile}
{signals}
{library_hint}
{ip_context}
{failure_context}

WEAK-MODEL SUPPORT PROTOCOL:
1. Do not guess the whole solution in one jump.
2. Pick one operator family from the UNIVERSAL INTERFACE PROFILE.
3. Decompose your hypothesis into typed holes: axis/context, variable(s),
   relation/operator, constants/thresholds, output schema.
4. Close each hole by a small executable measurement on the observed evidence.
5. Prefer short programs whose intermediate choices are explicit variables.
6. Never use axis-like fields (year/id/time columns) as scientific variables
   unless the task explicitly asks for that axis.

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

    # ----- Residual-to-operator self layers ------------------------------
    def _residual_operator_layer_sources(self) -> list[dict[str, Any]]:
        """Universal self-layer templates born from residual patterns.

        These are architecture search operators, not benchmark answers.  They
        inspect only the generic context and emit candidate programs/operators
        that are sandboxed and held-out scored before they can win.
        """

        suffix_layer_code = r'''
def build_layer(context: dict) -> dict:
    signature = str(context.get('signature_hint', ''))
    if 'current: str' not in signature or 'context: dict' not in signature:
        return {
            'operator_sources': [],
            'signals': [],
            'residual_hints': ['common_suffix_operator skipped: signature is not string transform'],
            'metadata': {'residual_operator': 'common_suffix'},
        }
    observations = list(context.get('observations') or [])
    if not observations:
        return {'operator_sources': [], 'signals': [], 'residual_hints': [], 'metadata': {'residual_operator': 'common_suffix'}}
    suffixes = []
    for row in observations:
        current = str(row.get('inputs', ''))
        target = str(row.get('target', ''))
        if not target.startswith(current):
            return {
                'operator_sources': [],
                'signals': ['targets do not share an input-prefix residual'],
                'residual_hints': ['try non-prefix string operators or context-gated composition'],
                'metadata': {'residual_operator': 'common_suffix', 'matched': False},
            }
        suffixes.append(target[len(current):])
    if not suffixes:
        return {'operator_sources': [], 'signals': [], 'residual_hints': [], 'metadata': {'residual_operator': 'common_suffix'}}
    suffix = suffixes[0]
    for item in suffixes:
        if item != suffix:
            return {
                'operator_sources': [],
                'signals': ['prefix residuals exist but suffixes differ'],
                'residual_hints': ['cluster suffixes by context before appending'],
                'metadata': {'residual_operator': 'common_suffix', 'matched': False},
            }
    escaped = suffix.replace('\\', '\\\\').replace("'", "\\'")
    operator_code = (
        "def transform(program_sources: list, context: dict) -> list:\n"
        "    observations = list(context.get('observations') or [])\n"
        "    suffixes = []\n"
        "    for row in observations:\n"
        "        current = str(row.get('inputs', ''))\n"
        "        target = str(row.get('target', ''))\n"
        "        if not target.startswith(current):\n"
        "            return []\n"
        "        suffixes.append(target[len(current):])\n"
        "    if not suffixes:\n"
        "        return []\n"
        "    suffix = suffixes[0]\n"
        "    for item in suffixes:\n"
        "        if item != suffix:\n"
        "            return []\n"
        "    escaped = suffix.replace('\\\\', '\\\\\\\\').replace(\"'\", \"\\\\'\")\n"
        "    literal = \"'\" + escaped + \"'\"\n"
        "    code = 'def rule(current: str, context: dict) -> str:\\n    return current + ' + literal + '\\n'\n"
        "    safe = ''\n"
        "    for ch in suffix:\n"
        "        if ch.isalnum():\n"
        "            safe = safe + ch.lower()\n"
        "    if not safe:\n"
        "        safe = 'suffix'\n"
        "    return [{\n"
        "        'name': 'residual_common_suffix_' + safe[:24],\n"
        "        'description': 'append the common target-minus-input suffix inferred from residuals',\n"
        "        'complexity': 1.2,\n"
        "        'code': code,\n"
        "    }]\n"
    )
    direct_code = "def rule(current: str, context: dict) -> str:\n    return current + '" + escaped + "'\n"
    return {
        'program_sources': [{
            'name': 'residual_direct_common_suffix',
            'description': 'direct program emitted from a common target-minus-input residual',
            'complexity': 1.1,
            'code': direct_code,
        }],
        'operator_sources': [{
            'name': 'residual_common_suffix_operator',
            'description': 'turn a common prefix residual into suffix-appending candidate programs',
            'code': operator_code,
        }],
        'signals': ['all targets equal input plus a shared suffix'],
        'residual_hints': ['common residual suffix=' + suffix],
        'metadata': {'residual_operator': 'common_suffix', 'matched': True},
    }
'''
        return [
            {
                "name": "residual_common_suffix_layer",
                "layer_type": "residual_operator_generator",
                "description": (
                    "Generic residual-to-operator layer: if string targets share "
                    "input as prefix, emit suffix repair candidates and an operator."
                ),
                "contract": {
                    "input": "context: dict",
                    "output": "dict(program_sources, operator_sources, signals, residual_hints, metadata)",
                    "scope": "generic_residual_pattern",
                },
                "code": suffix_layer_code.strip(),
            }
        ]

    def _propose_residual_operator_layers(
        self,
        adapter: CPIAdapter,
        observations: list[Observation],
        base_programs: list[HypothesisProgram],
        failure_context: str = "",
    ) -> tuple[list[HypothesisProgram], list[dict], list[str], list[dict[str, Any]]]:
        if (
            not self._propose_self_layers_enabled()
            or not self._residual_operator_layers_enabled()
            or not observations
        ):
            return [], [], [], []
        programs: list[HypothesisProgram] = []
        proposals: list[dict] = []
        errors: list[str] = []
        outputs: list[dict[str, Any]] = []
        for spec in self._residual_operator_layer_sources():
            code = str(spec.get("code", "")).strip()
            name = _safe_name(str(spec.get("name", "residual_operator_layer")))
            if not code:
                continue
            layer_programs, layer_errors, output = self._execute_layer_source(
                adapter,
                observations,
                layer_name=name,
                code=code,
                layer_origin="proposed",
                failure_context=failure_context,
                metadata=spec,
            )
            programs.extend(layer_programs)
            errors.extend(layer_errors)
            if output is None:
                continue
            outputs.append(output)
            proposals.append({"kind": "residual_operator_layer", **spec})
            operator_programs, operator_errors = self._programs_from_layer_operators(
                adapter,
                [output],
                base_programs + layer_programs,
                observations,
                failure_context=failure_context,
            )
            programs.extend(operator_programs)
            errors.extend(operator_errors)
        return programs, proposals, errors, outputs

    def _residual_kernel_programs(
        self,
        adapter: CPIAdapter,
        train_observations: list[Observation],
    ) -> tuple[list[HypothesisProgram], list[str], list[dict[str, Any]]]:
        """Use the residual kernel to induce rewrite-born programs.

        This is the bridge from the new microkernel into UniversalCPI.  It is
        still benchmark-agnostic: activation depends only on signature and
        observed Python types, and every emitted program is sandboxed and
        refuted before it can win.
        """

        if not train_observations:
            return [], [], []
        sig = adapter.signature_hint()
        induced: list[Any] = []
        proposals: list[dict[str, Any]] = []
        if (
            "inputs: dict" in sig
            and all(isinstance(o.inputs, dict) for o in train_observations)
            and all(isinstance(o.target, (int, float)) for o in train_observations)
        ):
            examples: list[dict[str, Any]] = []
            for obs in train_observations:
                row = {
                    str(k): v
                    for k, v in obs.inputs.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)
                }
                row["target"] = float(obs.target)
                examples.append(row)
            program = self._residual_kernel.induce_numeric_power_rule(examples)
            if program is not None:
                induced.append(program)
        if (
            "current: str" in sig
            and "context: dict" in sig
            and all(isinstance(o.inputs, str) and isinstance(o.target, str) for o in train_observations)
        ):
            examples = [
                {
                    "current": obs.inputs,
                    "target": obs.target,
                    "context": {
                        k: v
                        for k, v in obs.context.items()
                        if not k.startswith("__")
                    },
                }
                for obs in train_observations
            ]
            induced.extend(self._residual_kernel.induce_string_context_rules(examples))
        if (
            "analyze(df)" in sig
            and all(hasattr(o.inputs, "columns") for o in train_observations)
        ):
            induced.extend(
                self._residual_kernel.induce_table_evidence_programs(
                    question=str(getattr(adapter, "question", "")),
                    column_descriptions=dict(getattr(adapter, "column_descriptions", {}) or {}),
                    domain_context=str(getattr(adapter, "domain_knowledge", "")),
                )
            )

        programs: list[HypothesisProgram] = []
        errors: list[str] = []
        sg = adapter.sandbox_globals()
        for item in induced[:64]:
            code = str(item.code).strip()
            ok, err, fn = _sandbox_compile(code, sg)
            if not ok:
                errors.append(f"residual_kernel:{item.name}: {err}")
                continue
            program = HypothesisProgram(
                name=str(item.name),
                description=str(item.description),
                code=code,
                fn=fn,
                complexity=float(item.complexity),
                tags=("residual_kernel", item.rule.name),
            )
            programs.append(adapter.calibrate(program, train_observations))
            proposals.append(
                {
                    "kind": "residual_kernel",
                    "name": item.name,
                    "description": item.description,
                    "rule": item.rule.name,
                    "residual": item.rule.residual.kind,
                    "replace": dict(item.rule.replace),
                }
            )
        return programs, errors, proposals

    def _dpsr_programs(
        self,
        adapter: CPIAdapter,
        train_observations: list[Observation],
    ) -> tuple[list[HypothesisProgram], list[str], list[dict[str, Any]]]:
        """DPSR: Bayesian typed-hole induction as a pre-proposal layer.

        The LLM proposes only a hypothesis SKELETON with typed holes; the engine
        infers each hole's value by executable micro-measurements (posterior over
        candidates by held-in loss). Universal: drives only the adapter contract.
        Every emitted program is sandboxed and held-out scored like any other.
        """

        if not self._dpsr_enabled() or not train_observations or self.n_proposals <= 0:
            return [], [], []
        try:
            from mars.induction.dpsr import DPSREngine
        except Exception as exc:  # pragma: no cover - import guard
            return [], [f"dpsr: import failed: {exc}"], []
        engine = DPSREngine(
            model=self.model,
            client=self._client_lazy(),
            temperature=min(0.7, max(0.2, self.temperature)),
        )
        try:
            result = engine.run(adapter, train_observations)
        except Exception as exc:
            return [], [f"dpsr: run failed: {exc}"], []
        proposals: list[dict[str, Any]] = [
            {"kind": "dpsr", "name": p.name, "description": p.description}
            for p in result.programs
        ]
        if result.trace.steps:
            proposals.append({"kind": "dpsr_trace", "steps": result.trace.steps[:24]})
        return result.programs, result.errors, proposals

    def _nova_programs(
        self,
        adapter: CPIAdapter,
        train_observations: list[Observation],
    ) -> tuple[list[HypothesisProgram], list[str], list[dict[str, Any]]]:
        """NOVA: train a local hypothesis prior from oracle tasks.

        Unlike LLM proposals, NOVA first creates a small supervised curriculum
        from the current instance, updates a local prior over executable analyzer
        sketches, and only then emits programs.  The emitted programs remain
        ordinary CPI hypotheses: sandboxed, calibrated, and held-out scored.
        """

        if not self._nova_enabled() or not train_observations:
            return [], [], []
        try:
            from mars.nova import NOVASynthesizer
        except Exception as exc:  # pragma: no cover - import guard
            return [], [f"nova: import failed: {exc}"], []
        max_programs = int(os.environ.get("MARS_NOVA_MAX_PROGRAMS", "24"))
        temperature = float(os.environ.get("MARS_NOVA_TEMPERATURE", "0.25"))
        complexity_weight = float(os.environ.get("MARS_NOVA_COMPLEXITY_WEIGHT", "0.03"))
        synthesizer = NOVASynthesizer(
            max_programs=max_programs,
            temperature=temperature,
            complexity_weight=complexity_weight,
        )
        try:
            result = synthesizer.synthesize(
                signature_hint=adapter.signature_hint(),
                observations=train_observations,
                interface_name=adapter.name,
                question=str(getattr(adapter, "question", "")),
                column_descriptions=dict(getattr(adapter, "column_descriptions", {}) or {}),
                domain_context=str(getattr(adapter, "domain_knowledge", "")),
            )
        except Exception as exc:
            return [], [f"nova: synthesize failed: {exc}"], []
        programs: list[HypothesisProgram] = []
        errors: list[str] = []
        proposals: list[dict[str, Any]] = [
            {
                "kind": "nova_trace",
                "curriculum": dict(result.curriculum.features),
                "trace": list(result.trace)[:24],
            }
        ]
        sg = adapter.sandbox_globals()
        for item in result.program_sources:
            code = str(item.get("code", "")).strip()
            if not code:
                continue
            name = str(item.get("name", f"nova_program_{len(programs)}"))
            ok, err, fn = _sandbox_compile(code, sg)
            if not ok:
                errors.append(f"nova:{name}: {err}")
                continue
            try:
                complexity = float(item.get("complexity", 2.0))
            except Exception:
                complexity = 2.0
            program = HypothesisProgram(
                name=name,
                description=str(item.get("description", "NOVA local-oracle analyzer")),
                code=code,
                fn=fn,
                complexity=complexity,
                tags=("nova",),
            )
            programs.append(adapter.calibrate(program, train_observations))
            proposals.append(
                {
                    "kind": "nova",
                    "name": name,
                    "description": program.description,
                    "nova_loss": item.get("nova_loss"),
                    "nova_posterior": item.get("nova_posterior"),
                }
            )
        return programs, errors, proposals

    def _syndrome_programs(
        self,
        adapter: CPIAdapter,
        train_observations: list[Observation],
        base_programs: list[HypothesisProgram],
    ) -> tuple[list[HypothesisProgram], list[str], list[dict[str, Any]]]:
        """Decode missing hypothesis operators from candidate failure syndromes."""

        if not self._syndrome_enabled() or not train_observations:
            return [], [], []
        try:
            from mars.skills.syndrome_decoder import decode_syndrome_program_sources
        except Exception as exc:  # pragma: no cover - import guard
            return [], [f"syndrome: import failed: {exc}"], []
        try:
            decoded = decode_syndrome_program_sources(
                adapter=adapter,
                observations=train_observations,
                programs=base_programs,
            )
        except Exception as exc:
            return [], [f"syndrome: decode failed: {exc}"], []
        if decoded is None:
            return [], [], []
        programs: list[HypothesisProgram] = []
        errors: list[str] = []
        sg = adapter.sandbox_globals()
        for item in decoded.program_sources:
            code = str(item.get("code", "")).strip()
            name = str(item.get("name", f"syndrome_program_{len(programs)}"))
            if not code:
                continue
            ok, err, fn = _sandbox_compile(code, sg)
            if not ok:
                errors.append(f"syndrome:{name}: {err}")
                continue
            program = HypothesisProgram(
                name=name,
                description=str(item.get("description", decoded.rationale)),
                code=code,
                fn=fn,
                complexity=float(item.get("complexity", 3.0)),
                tags=("syndrome", decoded.decoded_operator),
            )
            programs.append(adapter.calibrate(program, train_observations))
        proposals = [
            {
                "kind": "syndrome_trace",
                "name": decoded.name,
                "decoded_operator": decoded.decoded_operator,
                "bits": dict(decoded.bits),
                "rationale": decoded.rationale,
            }
        ]
        for program in programs:
            proposals.append(
                {
                    "kind": "syndrome",
                    "name": program.name,
                    "description": program.description,
                    "decoded_operator": decoded.decoded_operator,
                }
            )
        return programs, errors, proposals

    def _darwin_programs(
        self,
        adapter: CPIAdapter,
        train_observations: list[Observation],
    ) -> tuple[list[HypothesisProgram], list[str], list[dict[str, Any]]]:
        """Evolve formal theory genomes and emit executable phenotypes."""

        if not self._darwin_enabled() or not train_observations:
            return [], [], []
        try:
            from mars.darwin import DarwinSynthesizer
            from mars.darwin.regulatory import RegulatoryGenome
        except Exception as exc:  # pragma: no cover - import guard
            return [], [f"darwin: import failed: {exc}"], []
        max_programs = int(os.environ.get("MARS_DARWIN_MAX_PROGRAMS", "32"))
        regulatory_genome = None
        genome_path = os.environ.get("MARS_REGULATORY_GENOME", "").strip()
        if genome_path:
            try:
                regulatory_genome = RegulatoryGenome.load_json(genome_path)
            except Exception as exc:
                return [], [f"darwin: regulatory genome load failed: {exc}"], []
        synthesizer = DarwinSynthesizer(
            max_programs=max_programs,
            regulatory_genome=regulatory_genome,
        )
        try:
            result = synthesizer.synthesize(
                signature_hint=adapter.signature_hint(),
                observations=train_observations,
                interface_name=adapter.name,
                question=str(getattr(adapter, "question", "")),
                column_descriptions=dict(getattr(adapter, "column_descriptions", {}) or {}),
                domain_context=str(getattr(adapter, "domain_knowledge", "")),
            )
        except Exception as exc:
            return [], [f"darwin: synthesize failed: {exc}"], []
        programs: list[HypothesisProgram] = []
        errors: list[str] = []
        proposals: list[dict[str, Any]] = [
            {
                "kind": "darwin_trace",
                "trace": list(result.trace)[:32],
            }
        ]
        sg = adapter.sandbox_globals()
        for item in result.program_sources:
            code = str(item.get("code", "")).strip()
            name = str(item.get("name", f"darwin_program_{len(programs)}"))
            if not code:
                continue
            ok, err, fn = _sandbox_compile(code, sg)
            if not ok:
                errors.append(f"darwin:{name}: {err}")
                continue
            program = HypothesisProgram(
                name=name,
                description=str(item.get("description", "Darwinian theory phenotype")),
                code=code,
                fn=fn,
                complexity=float(item.get("complexity", 3.0)),
                tags=("darwin",),
            )
            programs.append(adapter.calibrate(program, train_observations))
            proposals.append(
                {
                    "kind": "darwin",
                    "name": program.name,
                    "description": program.description,
                    "genome": item.get("genome"),
                    "lineage": item.get("lineage"),
                }
            )
        return programs, errors, proposals

    def _darwin_trace_only(
        self,
        adapter: CPIAdapter,
        train_observations: list[Observation],
    ) -> list[dict[str, Any]]:
        """Record regulatory development even when earlier priors solve a task."""

        if not self._darwin_enabled() or not train_observations:
            return []
        try:
            from mars.darwin import DarwinSynthesizer
            from mars.darwin.regulatory import RegulatoryGenome
        except Exception:
            return []
        regulatory_genome = None
        genome_path = os.environ.get("MARS_REGULATORY_GENOME", "").strip()
        if genome_path:
            try:
                regulatory_genome = RegulatoryGenome.load_json(genome_path)
            except Exception:
                regulatory_genome = None
        try:
            result = DarwinSynthesizer(
                max_programs=0,
                regulatory_genome=regulatory_genome,
            ).synthesize(
                signature_hint=adapter.signature_hint(),
                observations=train_observations,
                interface_name=adapter.name,
                question=str(getattr(adapter, "question", "")),
                column_descriptions=dict(getattr(adapter, "column_descriptions", {}) or {}),
                domain_context=str(getattr(adapter, "domain_knowledge", "")),
            )
        except Exception:
            return []
        return [
            {
                "kind": "darwin_trace",
                "trace_only": True,
                "trace": list(result.trace)[:4],
            }
        ]

    def _seed_programs(
        self,
        adapter: CPIAdapter,
        train_observations: list[Observation],
    ) -> tuple[list[HypothesisProgram], list[str]]:
        programs: list[HypothesisProgram] = []
        errors: list[str] = []
        sg = adapter.sandbox_globals()
        for i, item in enumerate(adapter.seed_program_sources()):
            if isinstance(item, str):
                code = item.strip()
                name = f"seed_{i}"
                description = "trusted seed program"
                complexity = 1.5
            elif isinstance(item, dict):
                code = str(item.get("code", "")).strip()
                name = str(item.get("name", f"seed_{i}"))
                description = str(item.get("description", "trusted seed program"))
                try:
                    complexity = float(item.get("complexity", 1.5))
                except Exception:
                    complexity = 1.5
            else:
                continue
            if not code:
                continue
            ok, err, fn = _sandbox_compile(code, sg)
            if not ok:
                errors.append(f"{name}: {err}")
                continue
            program = HypothesisProgram(
                name=name,
                description=description,
                code=code,
                fn=fn,
                complexity=complexity,
                tags=("trusted_seed",),
            )
            programs.append(adapter.calibrate(program, train_observations))
        return programs, errors

    def _typed_prior_programs(
        self,
        adapter: CPIAdapter,
        train_observations: list[Observation],
    ) -> tuple[list[HypothesisProgram], list[str]]:
        """Generate cheap type-directed program priors before LLM synthesis.

        These are universal templates over observed Python types, not benchmark
        answers.  They give the self-building layer a small executable basis
        that can later be rejected, repaired, composed, or branched.
        """

        if not train_observations:
            return [], []
        sig = adapter.signature_hint()
        if "current: str" not in sig or "context: dict" not in sig:
            if (
                "inputs: dict" in sig
                and all(isinstance(o.inputs, dict) for o in train_observations)
                and all(isinstance(o.target, (int, float)) for o in train_observations)
            ):
                return self._numeric_typed_prior_programs(adapter, train_observations)
            if (
                "analyze(df)" in sig
                and all(hasattr(o.inputs, "columns") for o in train_observations)
            ):
                return self._table_typed_prior_programs(adapter, train_observations)
            return [], []
        if not all(isinstance(o.inputs, str) and isinstance(o.target, str) for o in train_observations):
            return [], []

        context_string_keys: list[str] = []
        context_int_keys: list[str] = []
        for obs in train_observations:
            for key, value in obs.context.items():
                if key.startswith("__"):
                    continue
                if isinstance(value, str) and key not in context_string_keys:
                    context_string_keys.append(key)
                if isinstance(value, int) and key not in context_int_keys:
                    context_int_keys.append(key)

        def q(value: str) -> str:
            return repr(value)

        shift_helpers = (
            "def _shift_char(ch: str, delta: int) -> str:\n"
            "    if not ch:\n"
            "        return ch\n"
            "    o = ord(ch) - ord('A')\n"
            "    if o < 0 or o >= 26:\n"
            "        return ch\n"
            "    return chr(ord('A') + ((o + delta) % 26))\n\n"
            "def _shift(s: str, delta: int) -> str:\n"
            "    return ''.join(_shift_char(ch, delta) for ch in str(s))\n"
        )
        aggregate_helpers = (
            shift_helpers
            + "\n\ndef _is_prime(n: int) -> bool:\n"
            "    if n < 2:\n"
            "        return False\n"
            "    d = 2\n"
            "    while d * d <= n:\n"
            "        if n % d == 0:\n"
            "            return False\n"
            "        d += 1\n"
            "    return True\n\n"
            "def _most_frequent_char(s: str) -> str:\n"
            "    counts = {}\n"
            "    for ch in str(s):\n"
            "        counts[ch] = counts.get(ch, 0) + 1\n"
            "    if not counts:\n"
            "        return ''\n"
            "    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]\n"
        )
        sources: list[dict[str, Any]] = [
            {
                "name": "typed_identity",
                "description": "return the current string unchanged",
                "complexity": 1.0,
                "code": "def rule(current: str, context: dict) -> str:\n    return current\n",
            },
            {
                "name": "typed_reverse_current",
                "description": "reverse the current string",
                "complexity": 1.2,
                "code": "def rule(current: str, context: dict) -> str:\n    return current[::-1]\n",
            },
        ]

        for key in context_string_keys[:4]:
            sources.extend([
                {
                    "name": f"typed_context_{key}",
                    "description": f"return context string {key}",
                    "complexity": 1.2,
                    "code": (
                        "def rule(current: str, context: dict) -> str:\n"
                        f"    return str(context.get({q(key)}, ''))\n"
                    ),
                },
                {
                    "name": f"typed_current_plus_{key}",
                    "description": f"append context string {key} to current",
                    "complexity": 1.5,
                    "code": (
                        "def rule(current: str, context: dict) -> str:\n"
                        f"    return current + str(context.get({q(key)}, ''))\n"
                    ),
                },
                {
                    "name": f"typed_{key}_plus_current",
                    "description": f"prepend context string {key} to current",
                    "complexity": 1.5,
                    "code": (
                        "def rule(current: str, context: dict) -> str:\n"
                        f"    return str(context.get({q(key)}, '')) + current\n"
                    ),
                },
            ])

        for a_i, a_key in enumerate(context_string_keys[:4]):
            for b_key in context_string_keys[a_i + 1:4]:
                sources.extend([
                    {
                        "name": f"typed_interleave_{a_key}_{b_key}",
                        "description": f"interleave context strings {a_key} and {b_key}",
                        "complexity": 2.0,
                        "code": (
                            "def rule(current: str, context: dict) -> str:\n"
                            f"    a = str(context.get({q(a_key)}, ''))\n"
                            f"    b = str(context.get({q(b_key)}, ''))\n"
                            "    out = []\n"
                            "    for x, y in zip(a, b):\n"
                            "        out.append(x)\n"
                            "        out.append(y)\n"
                            "    return ''.join(out)\n"
                        ),
                    },
                    {
                        "name": f"typed_interleave_{b_key}_{a_key}",
                        "description": f"interleave context strings {b_key} and {a_key}",
                        "complexity": 2.0,
                        "code": (
                            "def rule(current: str, context: dict) -> str:\n"
                            f"    a = str(context.get({q(a_key)}, ''))\n"
                            f"    b = str(context.get({q(b_key)}, ''))\n"
                            "    out = []\n"
                            "    for x, y in zip(a, b):\n"
                            "        out.append(y)\n"
                            "        out.append(x)\n"
                            "    return ''.join(out)\n"
                        ),
                    },
                ])

        for n_key in context_int_keys[:3]:
            sources.extend([
                {
                    "name": f"typed_shift_current_by_{n_key}",
                    "description": f"cyclically shift current characters by {n_key}",
                    "complexity": 2.0,
                    "code": (
                        shift_helpers
                        + "\n\ndef rule(current: str, context: dict) -> str:\n"
                        f"    n = int(context.get({q(n_key)}, 0))\n"
                        "    return _shift(current, n)\n"
                    ),
                },
                {
                    "name": f"typed_reverse_shift_concat_shift_{n_key}",
                    "description": f"reverse current, shift by {n_key}, then append shifted current",
                    "complexity": 2.6,
                    "code": (
                        shift_helpers
                        + "\n\ndef rule(current: str, context: dict) -> str:\n"
                        f"    n = int(context.get({q(n_key)}, 0))\n"
                        "    return _shift(current[::-1], n) + _shift(current, n)\n"
                    ),
                },
                {
                    "name": f"typed_repeat_indexed_char_by_{n_key}",
                    "description": f"append one indexed current character repeated {n_key} times",
                    "complexity": 2.3,
                    "code": (
                        "def rule(current: str, context: dict) -> str:\n"
                        "    if not current:\n"
                        "        return current\n"
                        f"    n = int(context.get({q(n_key)}, 0))\n"
                        "    if n <= 0:\n"
                        "        return current\n"
                        "    ch = current[(n - 1) % len(current)]\n"
                        "    return current + ch * n\n"
                    ),
                },
                {
                    "name": f"typed_replace_most_frequent_on_prime_{n_key}",
                    "description": (
                        f"on prime {n_key}, replace every most frequent current "
                        "character with its next cyclic character"
                    ),
                    "complexity": 3.0,
                    "code": (
                        aggregate_helpers
                        + "\n\ndef rule(current: str, context: dict) -> str:\n"
                        f"    n = int(context.get({q(n_key)}, 0))\n"
                        "    if not _is_prime(n):\n"
                        "        return current\n"
                        "    ch = _most_frequent_char(current)\n"
                        "    if not ch:\n"
                        "        return current\n"
                        "    repl = _shift_char(ch, 1)\n"
                        "    return ''.join(repl if c == ch else c for c in current)\n"
                    ),
                },
                {
                    "name": f"typed_replace_most_frequent_always_{n_key}",
                    "description": (
                        "replace every most frequent current character with its "
                        f"next cyclic character, ignoring {n_key}"
                    ),
                    "complexity": 2.8,
                    "code": (
                        aggregate_helpers
                        + "\n\ndef rule(current: str, context: dict) -> str:\n"
                        "    ch = _most_frequent_char(current)\n"
                        "    if not ch:\n"
                        "        return current\n"
                        "    repl = _shift_char(ch, 1)\n"
                        "    return ''.join(repl if c == ch else c for c in current)\n"
                    ),
                },
            ])
            for key in context_string_keys[:4]:
                sources.append(
                    {
                        "name": f"typed_shift_{key}_by_{n_key}",
                        "description": f"cyclically shift context string {key} by {n_key}",
                        "complexity": 2.0,
                        "code": (
                            shift_helpers
                            + "\n\ndef rule(current: str, context: dict) -> str:\n"
                            f"    n = int(context.get({q(n_key)}, 0))\n"
                            f"    return _shift(str(context.get({q(key)}, '')), n)\n"
                        ),
                    }
                )

        for key in context_string_keys[:4]:
            sources.append(
                {
                    "name": f"typed_charwise_add_current_{key}",
                    "description": f"charwise modular addition of current and {key}",
                    "complexity": 2.8,
                    "code": (
                        "def rule(current: str, context: dict) -> str:\n"
                        f"    other = str(context.get({q(key)}, ''))\n"
                        "    out = []\n"
                        "    for i, ch in enumerate(current):\n"
                        "        if i >= len(other):\n"
                        "            out.append(ch)\n"
                        "            continue\n"
                        "        a = ord(ch) - ord('A')\n"
                        "        b = ord(other[i]) - ord('A')\n"
                        "        if 0 <= a < 26 and 0 <= b < 26:\n"
                        "            out.append(chr(ord('A') + ((a + b) % 26)))\n"
                        "        else:\n"
                        "            out.append(ch)\n"
                        "    return ''.join(out)\n"
                    ),
                }
            )

        programs: list[HypothesisProgram] = []
        errors: list[str] = []
        sg = adapter.sandbox_globals()
        for item in sources:
            ok, err, fn = _sandbox_compile(str(item["code"]), sg)
            if not ok:
                errors.append(f"{item['name']}: {err}")
                continue
            programs.append(
                adapter.calibrate(
                    HypothesisProgram(
                        name=str(item["name"]),
                        description=str(item["description"]),
                        code=str(item["code"]),
                        fn=fn,
                        complexity=float(item["complexity"]),
                        tags=("typed_prior",),
                    ),
                    train_observations,
                )
            )
        return programs, errors

    def _numeric_typed_prior_programs(
        self,
        adapter: CPIAdapter,
        train_observations: list[Observation],
    ) -> tuple[list[HypothesisProgram], list[str]]:
        """Generate type-directed numeric law structures for dict -> scalar tasks.

        This is deliberately generic: it knows only that inputs are numeric
        fields and targets are numbers.  Multiplicative constants are left for
        the adapter's data-driven calibration step.
        """

        keys: list[str] = []
        for obs in train_observations:
            for key, value in obs.inputs.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if key not in keys:
                        keys.append(str(key))
        keys = keys[:8]
        if not keys:
            return [], []

        def q(value: str) -> str:
            return repr(value)

        helpers = (
            "def _v(inputs: dict, key: str) -> float:\n"
            "    return float(inputs.get(key, 0.0))\n\n"
            "def _safe_div(a: float, b: float) -> float:\n"
            "    if abs(b) < 1e-12:\n"
            "        return 0.0\n"
            "    return a / b\n"
        )
        sources: list[dict[str, Any]] = [
            {
                "name": "typed_numeric_constant",
                "description": "constant numeric structure",
                "complexity": 1.0,
                "code": "def law(inputs: dict) -> float:\n    return 1.0\n",
            }
        ]
        for key in keys:
            sources.extend(
                [
                    {
                        "name": f"typed_numeric_{key}",
                        "description": f"use numeric variable {key}",
                        "complexity": 1.2,
                        "code": (
                            "def law(inputs: dict) -> float:\n"
                            f"    return float(inputs.get({q(key)}, 0.0))\n"
                        ),
                    },
                    {
                        "name": f"typed_numeric_square_{key}",
                        "description": f"use square of numeric variable {key}",
                        "complexity": 1.6,
                        "code": (
                            "def law(inputs: dict) -> float:\n"
                            f"    x = float(inputs.get({q(key)}, 0.0))\n"
                            "    return x * x\n"
                        ),
                    },
                    {
                        "name": f"typed_numeric_inverse_square_{key}",
                        "description": f"use inverse square of numeric variable {key}",
                        "complexity": 1.8,
                        "code": (
                            helpers
                            + "\n\ndef law(inputs: dict) -> float:\n"
                            f"    x = _v(inputs, {q(key)})\n"
                            "    return _safe_div(1.0, x * x)\n"
                        ),
                    },
                    {
                        "name": f"typed_numeric_sqrt_{key}",
                        "description": f"use square root magnitude of numeric variable {key}",
                        "complexity": 1.8,
                        "code": (
                            "def law(inputs: dict) -> float:\n"
                            f"    x = float(inputs.get({q(key)}, 0.0))\n"
                            "    return math.sqrt(abs(x))\n"
                        ),
                    },
                ]
            )

        for i, a_key in enumerate(keys):
            for b_key in keys[i + 1:]:
                sources.extend(
                    [
                        {
                            "name": f"typed_numeric_product_{a_key}_{b_key}",
                            "description": f"use product of numeric variables {a_key} and {b_key}",
                            "complexity": 2.0,
                            "code": (
                                helpers
                                + "\n\ndef law(inputs: dict) -> float:\n"
                                f"    return _v(inputs, {q(a_key)}) * _v(inputs, {q(b_key)})\n"
                            ),
                        },
                        {
                            "name": f"typed_numeric_ratio_{a_key}_over_{b_key}",
                            "description": f"use ratio {a_key} / {b_key}",
                            "complexity": 2.0,
                            "code": (
                                helpers
                                + "\n\ndef law(inputs: dict) -> float:\n"
                                f"    return _safe_div(_v(inputs, {q(a_key)}), _v(inputs, {q(b_key)}))\n"
                            ),
                        },
                        {
                            "name": f"typed_numeric_ratio_{b_key}_over_{a_key}",
                            "description": f"use ratio {b_key} / {a_key}",
                            "complexity": 2.0,
                            "code": (
                                helpers
                                + "\n\ndef law(inputs: dict) -> float:\n"
                                f"    return _safe_div(_v(inputs, {q(b_key)}), _v(inputs, {q(a_key)}))\n"
                            ),
                        },
                        {
                            "name": f"typed_numeric_inverse_square_product_{a_key}_{b_key}",
                            "description": f"use {a_key} divided by square of {b_key}",
                            "complexity": 2.4,
                            "code": (
                                helpers
                                + "\n\ndef law(inputs: dict) -> float:\n"
                                f"    return _safe_div(_v(inputs, {q(a_key)}), _v(inputs, {q(b_key)}) ** 2)\n"
                            ),
                        },
                        {
                            "name": f"typed_numeric_inverse_square_product_{b_key}_{a_key}",
                            "description": f"use {b_key} divided by square of {a_key}",
                            "complexity": 2.4,
                            "code": (
                                helpers
                                + "\n\ndef law(inputs: dict) -> float:\n"
                                f"    return _safe_div(_v(inputs, {q(b_key)}), _v(inputs, {q(a_key)}) ** 2)\n"
                            ),
                        },
                        {
                            "name": f"typed_numeric_sum_{a_key}_{b_key}",
                            "description": f"use sum of numeric variables {a_key} and {b_key}",
                            "complexity": 1.8,
                            "code": (
                                helpers
                                + "\n\ndef law(inputs: dict) -> float:\n"
                                f"    return _v(inputs, {q(a_key)}) + _v(inputs, {q(b_key)})\n"
                            ),
                        },
                        {
                            "name": f"typed_numeric_difference_{a_key}_{b_key}",
                            "description": f"use difference {a_key} - {b_key}",
                            "complexity": 1.8,
                            "code": (
                                helpers
                                + "\n\ndef law(inputs: dict) -> float:\n"
                                f"    return _v(inputs, {q(a_key)}) - _v(inputs, {q(b_key)})\n"
                            ),
                        },
                    ]
                )
                for denom_key in keys:
                    if denom_key in (a_key, b_key):
                        continue
                    sources.append(
                        {
                            "name": f"typed_numeric_product_{a_key}_{b_key}_over_{denom_key}_square",
                            "description": (
                                f"use product of {a_key} and {b_key} divided by "
                                f"square of {denom_key}"
                            ),
                            "complexity": 3.0,
                            "code": (
                                helpers
                                + "\n\ndef law(inputs: dict) -> float:\n"
                                f"    num = _v(inputs, {q(a_key)}) * _v(inputs, {q(b_key)})\n"
                                f"    den = _v(inputs, {q(denom_key)}) ** 2\n"
                                "    return _safe_div(num, den)\n"
                            ),
                        }
                    )

        angle_keys = [k for k in keys if _angle_like_key(k)]
        ratio_keys = [k for k in keys if k not in set(angle_keys)]
        for angle_key in angle_keys[:3]:
            for numerator in ratio_keys[:5]:
                for denominator in ratio_keys[:5]:
                    if numerator == denominator:
                        continue
                    sources.append(
                        {
                            "name": f"typed_numeric_bounded_trig_{numerator}_over_{denominator}_sin_{angle_key}",
                            "description": (
                                "bounded angle relation: asin((ratio) * sin(angle)); "
                                "activated by angle-like interface variables"
                            ),
                            "complexity": 2.7,
                            "code": (
                                helpers
                                + "\n\ndef law(inputs: dict) -> float:\n"
                                f"    den = _v(inputs, {q(denominator)})\n"
                                "    if abs(den) < 1e-12:\n"
                                "        return 0.0\n"
                                f"    value = (_v(inputs, {q(numerator)}) / den) * math.sin(math.radians(_v(inputs, {q(angle_key)})))\n"
                                "    if value > 1.0:\n"
                                "        value = 1.0\n"
                                "    if value < -1.0:\n"
                                "        value = -1.0\n"
                                "    return math.degrees(math.asin(value))\n"
                            ),
                        }
                    )
                    sources.append(
                        {
                            "name": f"typed_numeric_bounded_trig_acos_{numerator}_over_{denominator}_sin_{angle_key}",
                            "description": (
                                "bounded complementary angle relation: acos((ratio) * sin(angle)); "
                                "activated by angle-like interface variables"
                            ),
                            "complexity": 2.8,
                            "code": (
                                helpers
                                + "\n\ndef law(inputs: dict) -> float:\n"
                                f"    den = _v(inputs, {q(denominator)})\n"
                                "    if abs(den) < 1e-12:\n"
                                "        return 0.0\n"
                                f"    value = (_v(inputs, {q(numerator)}) / den) * math.sin(math.radians(_v(inputs, {q(angle_key)})))\n"
                                "    return math.degrees(math.acos(value))\n"
                            ),
                        }
                    )

        programs: list[HypothesisProgram] = []
        errors: list[str] = []
        sg = adapter.sandbox_globals()
        for item in sources:
            ok, err, fn = _sandbox_compile(str(item["code"]), sg)
            if not ok:
                errors.append(f"{item['name']}: {err}")
                continue
            programs.append(
                adapter.calibrate(
                    HypothesisProgram(
                        name=str(item["name"]),
                        description=str(item["description"]),
                        code=str(item["code"]),
                        fn=fn,
                        complexity=float(item["complexity"]),
                        tags=("typed_numeric_prior",),
                    ),
                    train_observations,
                )
            )
        return programs, errors

    def _table_typed_prior_programs(
        self,
        adapter: CPIAdapter,
        train_observations: list[Observation],
    ) -> tuple[list[HypothesisProgram], list[str]]:
        """Generate generic dataframe analyzer priors for table -> dict tasks."""

        first_df = train_observations[0].inputs
        columns = [str(c) for c in getattr(first_df, "columns", [])][:12]
        numeric_columns: list[str] = []
        for col in columns:
            try:
                series = first_df[col]
                if getattr(series, "dtype", None) is not None and str(series.dtype) != "object":
                    numeric_columns.append(col)
            except Exception:
                continue
        numeric_columns = numeric_columns[:8]

        def q(value: str) -> str:
            return repr(value)

        sources: list[dict[str, Any]] = [
            {
                "name": "typed_table_shape",
                "description": "summarize dataframe shape",
                "complexity": 1.0,
                "code": (
                    "def analyze(df) -> dict:\n"
                    "    return {'evidence': f'rows={len(df)}, columns={len(df.columns)}', "
                    "'statistic': float(len(df))}\n"
                ),
            }
        ]
        for col in numeric_columns:
            sources.extend(
                [
                    {
                        "name": f"typed_table_mean_{col}",
                        "description": f"mean statistic for numeric column {col}",
                        "complexity": 1.4,
                        "code": (
                            "def analyze(df) -> dict:\n"
                            f"    s = df[{q(col)}]\n"
                            "    v = float(s.mean())\n"
                            f"    return {{'evidence': 'mean({col})=' + str(v), 'statistic': v}}\n"
                        ),
                    },
                    {
                        "name": f"typed_table_std_{col}",
                        "description": f"standard deviation statistic for numeric column {col}",
                        "complexity": 1.6,
                        "code": (
                            "def analyze(df) -> dict:\n"
                            f"    s = df[{q(col)}]\n"
                            "    v = float(s.std())\n"
                            f"    return {{'evidence': 'std({col})=' + str(v), 'statistic': v}}\n"
                        ),
                    },
                ]
            )
        for i, a_col in enumerate(numeric_columns):
            for b_col in numeric_columns[i + 1:]:
                sources.append(
                    {
                        "name": f"typed_table_corr_{a_col}_{b_col}",
                        "description": f"correlation statistic between {a_col} and {b_col}",
                        "complexity": 2.2,
                        "code": (
                            "def analyze(df) -> dict:\n"
                            f"    v = float(df[{q(a_col)}].corr(df[{q(b_col)}]))\n"
                            f"    return {{'evidence': 'corr({a_col},{b_col})=' + str(v), 'statistic': v}}\n"
                        ),
                    }
                )

        programs: list[HypothesisProgram] = []
        errors: list[str] = []
        sg = adapter.sandbox_globals()
        for item in sources:
            ok, err, fn = _sandbox_compile(str(item["code"]), sg)
            if not ok:
                errors.append(f"{item['name']}: {err}")
                continue
            programs.append(
                adapter.calibrate(
                    HypothesisProgram(
                        name=str(item["name"]),
                        description=str(item["description"]),
                        code=str(item["code"]),
                        fn=fn,
                        complexity=float(item["complexity"]),
                        tags=("typed_table_prior",),
                    ),
                    train_observations,
                )
            )
        return programs, errors

    def _compose_seed_programs(
        self,
        adapter: CPIAdapter,
        seeds: list[HypothesisProgram],
        train_observations: list[Observation],
        *,
        max_compositions: int = 24,
    ) -> tuple[list[HypothesisProgram], list[str]]:
        if not seeds or not adapter.compose_seed_programs():
            return [], []
        programs: list[HypothesisProgram] = []
        errors: list[str] = []
        sg = adapter.sandbox_globals()
        count = 0
        for left in seeds:
            for right in seeds:
                if count >= max_compositions:
                    return programs, errors
                code = _compose_transform_sources(
                    left.code,
                    right.code,
                    left_name=f"_seed_left_{count}",
                    right_name=f"_seed_right_{count}",
                )
                if not code:
                    continue
                name = f"compose_{left.name}_then_{right.name}"[:96]
                ok, err, fn = _sandbox_compile(code, sg)
                if not ok:
                    errors.append(f"{name}: {err}")
                    continue
                count += 1
                programs.append(
                    adapter.calibrate(
                        HypothesisProgram(
                            name=name,
                            description=f"composition: {left.description} then {right.description}",
                            code=code,
                            fn=fn,
                            complexity=left.complexity + right.complexity + 0.5,
                            tags=("trusted_composition",),
                        ),
                        train_observations,
                    )
                )
        return programs, errors

    def _branch_seed_programs(
        self,
        adapter: CPIAdapter,
        seeds: list[HypothesisProgram],
        train_observations: list[Observation],
        *,
        max_branches: int = 160,
    ) -> tuple[list[HypothesisProgram], list[str]]:
        if not seeds or not adapter.branch_seed_programs():
            return [], []
        predicates = adapter.branch_predicate_sources(train_observations)
        if not predicates:
            return [], []

        ranked_seeds = self._rank(seeds, train_observations, adapter)
        seeds = [program for program, _score in ranked_seeds[:24]]
        programs: list[HypothesisProgram] = []
        errors: list[str] = []
        sg = adapter.sandbox_globals()
        count = 0
        for pred in predicates:
            pred_name = _safe_name(str(pred.get("name", f"predicate_{count}")))
            pred_expr = str(pred.get("expr", "")).strip()
            if not pred_expr:
                continue
            helper_source = str(pred.get("helper", "") or "")
            for true_seed in seeds:
                for false_seed in seeds:
                    if true_seed.name == false_seed.name:
                        continue
                    if count >= max_branches:
                        return programs, errors
                    code = _branch_transform_sources(
                        true_seed.code,
                        false_seed.code,
                        predicate_expr=pred_expr,
                        true_name=f"_seed_true_{count}",
                        false_name=f"_seed_false_{count}",
                        helper_source=helper_source,
                    )
                    if not code:
                        continue
                    name = (
                        f"branch_{pred_name}_{true_seed.name}_else_{false_seed.name}"
                    )[:160]
                    ok, err, fn = _sandbox_compile(code, sg)
                    if not ok:
                        errors.append(f"{name}: {err}")
                        continue
                    count += 1
                    programs.append(
                        adapter.calibrate(
                            HypothesisProgram(
                                name=name,
                                description=(
                                    f"branch on {pred_name}: if true use "
                                    f"{true_seed.description}; otherwise use "
                                    f"{false_seed.description}"
                                ),
                                code=code,
                                fn=fn,
                                complexity=(
                                    true_seed.complexity
                                    + false_seed.complexity
                                    + 1.0
                                ),
                                tags=("trusted_branch_composition",),
                            ),
                            train_observations,
                        )
                    )
        return programs, errors

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
            exact += int(l <= _ZERO_LOSS_EPS)
            losses.append(l)
        mean = sum(losses) / len(losses) if losses else 1.0
        evidence_bonus = self._evidence_structure_bonus(program, observations, adapter)
        evidence_bonus += self._numeric_interface_theory_bonus(program, observations, adapter)
        degeneracy_penalty = self._numeric_degeneracy_penalty(program, observations, adapter)
        return ProgramScore(
            name=program.name,
            loss_mean=mean,
            exact_rate=exact / len(losses) if losses else 0.0,
            mdl_score=mean + self.complexity_weight * program.complexity - evidence_bonus + degeneracy_penalty,
            n_scored=len(losses),
            complexity=program.complexity,
        )

    def _evidence_structure_bonus(
        self,
        program: HypothesisProgram,
        observations: list[Observation],
        adapter: CPIAdapter,
    ) -> float:
        """Reward structured evidence for table analyzers.

        Table benchmarks can otherwise be gamed by any analyzer that merely
        returns a finite number.  This bonus is still benchmark-agnostic: it
        looks for reusable evidence operators in the analyzer output, not for a
        dataset name or answer.
        """

        if not self._env_enabled("MARS_TABLE_EVIDENCE_BONUS", "0"):
            return 0.0
        if "analyze(df)" not in adapter.signature_hint() or not observations:
            return 0.0
        try:
            out = program.fn(observations[0].inputs)
        except Exception:
            return 0.0
        if not isinstance(out, dict):
            return 0.0
        evidence = str(out.get("evidence", "")).lower()
        try:
            statistic = abs(float(out.get("statistic", 0.0)))
        except Exception:
            statistic = 0.0
        bonus = 0.0
        if "corr(" in evidence:
            bonus += 0.025
        if "trend(" in evidence:
            bonus += 0.02
        if "mediator" in evidence or "mediation" in evidence:
            bonus += 0.02
        if "wide-mode" in evidence or "column-mode" in evidence:
            bonus += 0.015
        if statistic > 0:
            bonus += 0.005
        return min(0.08, bonus)

    def _numeric_interface_theory_bonus(
        self,
        program: HypothesisProgram,
        observations: list[Observation],
        adapter: CPIAdapter,
    ) -> float:
        """Prefer non-degenerate theory priors when numeric probes are uninformative."""

        if not _uninformative_angle_numeric_task(adapter, observations):
            return 0.0
        text = f"{program.name}\n{program.description}\n{program.code}".lower()
        ordered_bonus = _ordered_ratio_prior_bonus(program.name)
        if ("asin" in text or "acos" in text) and "sin" in text and "trig" in text:
            return 0.75 + ordered_bonus
        if "bounded_sine_ratio" in text or "bounded angle" in text:
            return 0.75 + ordered_bonus
        return 0.0

    def _numeric_degeneracy_penalty(
        self,
        program: HypothesisProgram,
        observations: list[Observation],
        adapter: CPIAdapter,
    ) -> float:
        if not _uninformative_angle_numeric_task(adapter, observations):
            return 0.0
        text = f"{program.name}\n{program.description}\n{program.code}".lower()
        if ("asin" in text or "acos" in text) and "sin" in text and ("trig" in text or "bounded" in text):
            return 0.0
        if _program_is_degenerate_constant(program, observations):
            return 1.0
        return 0.55

    def _rank(
        self,
        programs: list[HypothesisProgram],
        observations: list[Observation],
        adapter: CPIAdapter,
    ) -> list[tuple[HypothesisProgram, ProgramScore]]:
        ranked = [(p, self._score(p, observations, adapter)) for p in programs]
        ranked.sort(key=self._rank_key)
        return ranked

    @classmethod
    def _rank_key(
        cls,
        item: tuple[HypothesisProgram, ProgramScore],
    ) -> tuple[float, float, float, float]:
        program, score = item
        if score.loss_mean <= _ZERO_LOSS_EPS:
            return (
                0.0,
                float(cls._rank_tag_priority(program)),
                score.loss_mean,
                score.mdl_score,
            )
        return (
            1.0,
            score.loss_mean,
            score.mdl_score,
            program.complexity,
        )

    @staticmethod
    def _rank_tag_priority(program: HypothesisProgram) -> int:
        """Break exact ties without letting archived duplicates hide core seeds."""

        tags = set(program.tags)
        if "trusted_branch_composition" in tags:
            return 0
        if "trusted_composition" in tags:
            return 1
        if "nova" in tags:
            return 2
        if "self_layer_program" in tags:
            return 4
        return 3

    def _failure_examples(
        self,
        program: HypothesisProgram,
        observations: list[Observation],
        adapter: CPIAdapter,
        *,
        max_examples: int = 4,
    ) -> str:
        """Render concrete counterexamples for the next synthesis round."""

        rows: list[dict[str, Any]] = []
        for obs in observations:
            try:
                pred = adapter.execute(program, obs)
                loss = adapter.loss(pred, obs)
            except Exception as exc:
                pred = f"ERROR: {type(exc).__name__}: {exc}"
                loss = 1.0
            if loss <= _ZERO_LOSS_EPS:
                continue
            rows.append(
                {
                    "inputs": _truncate_repr(obs.inputs, 300),
                    "context": {
                        k: _truncate_repr(v, 180)
                        for k, v in obs.context.items()
                        if not k.startswith("__")
                    },
                    "prediction": _truncate_repr(pred, 300),
                    "target": _truncate_repr(obs.target, 300),
                    "loss": round(float(loss), 3) if loss == loss else 1.0,
                }
            )
            if len(rows) >= max_examples:
                break
        if not rows:
            return ""
        structural = self._structural_residual_context(program, observations, adapter)
        return (
            "\nCONCRETE COUNTEREXAMPLES FOR THE PREVIOUS BEST PROGRAM:\n"
            + json.dumps(rows, indent=2, ensure_ascii=False, default=str)
            + "\nRepair these exact residuals. Propose programs that explain "
            "the target, not just the previous prediction."
            + structural
        )

    def _structural_residual_context(
        self,
        program: HypothesisProgram,
        observations: list[Observation],
        adapter: CPIAdapter,
        *,
        max_residuals: int = 6,
    ) -> str:
        """Compile type/shape/schema residuals from observations.

        This is intentionally benchmark-agnostic.  It lets the next repair
        round know whether the failure is numeric, categorical, schema, shape,
        execution, or type-level without adding benchmark-specific rules.
        """

        try:
            from mars.skills.metric_compiler import (
                compile_interface_contract,
                score_interface_contract,
            )
        except Exception:
            return ""
        try:
            contract = compile_interface_contract(
                observations,
                name=f"{adapter.name}_interface_contract",
            )

            def predict(inputs: Any, context: dict[str, Any]) -> Any:
                obs = Observation(inputs=inputs, target=None, context=dict(context))
                return adapter.execute(program, obs)

            score = score_interface_contract(contract, observations, predict)
        except Exception:
            return ""
        if not score.residuals:
            return ""
        rows = [
            {
                "check": residual.check,
                "residual_type": residual.residual_type,
                "loss": round(float(residual.loss), 3),
                "prediction": residual.prediction_preview,
                "target": residual.target_preview,
                "rationale": residual.rationale,
            }
            for residual in score.residuals[:max_residuals]
        ]
        return (
            "\n\nSTRUCTURAL INTERFACE RESIDUALS "
            f"(contract={contract.name}, target_type={contract.target_type}):\n"
            + json.dumps(rows, indent=2, ensure_ascii=False, default=str)
        )

    # ----- Main loop ------------------------------------------------------
    def run(self, adapter: CPIAdapter) -> CPIResult:
        t0 = time.time()
        self._runtime_layer_sources = {}
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
        ip_context = self._ip_genome_context(adapter, train_obs)
        if ip_context:
            all_proposals.append({"kind": "ip_genome_context", **ip_context})

        layer_programs, layer_errors, layer_outputs = self._load_self_layer_programs(
            adapter,
            train_obs,
            failure_context,
        )
        all_programs.extend(layer_programs)
        all_errors.extend(layer_errors)
        if layer_outputs:
            layer_signal_rows = []
            for output in layer_outputs[:6]:
                layer_signal_rows.append(
                    {
                        "layer": output.get("_layer_name"),
                        "signals": list(output.get("signals") or [])[:4],
                        "residual_hints": list(output.get("residual_hints") or [])[:4],
                        "n_program_sources": len(list(output.get("program_sources") or [])),
                    }
                )
            failure_context += (
                "\nSELF-WRITTEN ARCHITECTURE LAYERS RAN BEFORE THIS ROUND:\n"
                + json.dumps(layer_signal_rows, indent=2, ensure_ascii=False, default=str)
            )
        self_modules, self_module_errors = self._load_self_module_programs(
            adapter,
            train_obs,
        )
        all_programs.extend(self_modules)
        all_errors.extend(self_module_errors)
        seeded, seed_errors = self._seed_programs(adapter, train_obs)
        all_programs.extend(seeded)
        all_errors.extend(seed_errors)
        typed_priors, typed_errors = (
            self._typed_prior_programs(adapter, train_obs)
            if self._typed_priors_enabled()
            else ([], [])
        )
        all_programs.extend(typed_priors)
        all_errors.extend(typed_errors)
        residual_kernel_programs, residual_kernel_errors, residual_kernel_proposals = (
            self._residual_kernel_programs(adapter, train_obs)
            if self._residual_kernel_enabled()
            else ([], [], [])
        )
        all_programs.extend(residual_kernel_programs)
        all_errors.extend(residual_kernel_errors)
        all_proposals.extend(residual_kernel_proposals)
        # DPSR: Bayesian typed-hole induction (skeleton from LLM, hole values
        # inferred by executable measurement). Sits between cheap priors and the
        # full LLM program proposal; every program is still held-out scored.
        dpsr_programs, dpsr_errors, dpsr_proposals = self._dpsr_programs(adapter, train_obs)
        all_programs.extend(dpsr_programs)
        all_errors.extend(dpsr_errors)
        all_proposals.extend(dpsr_proposals)
        operator_programs, operator_errors = self._programs_from_layer_operators(
            adapter,
            layer_outputs,
            layer_programs + self_modules + seeded + typed_priors + residual_kernel_programs + dpsr_programs,
            train_obs,
            failure_context=failure_context,
        )
        all_programs.extend(operator_programs)
        all_errors.extend(operator_errors)
        seed_pool = (
            layer_programs
            + self_modules
            + seeded
            + typed_priors
            + residual_kernel_programs
            + dpsr_programs
            + operator_programs
        )
        composed, composition_errors = self._compose_seed_programs(adapter, seed_pool, train_obs)
        all_programs.extend(composed)
        all_errors.extend(composition_errors)
        branched, branch_errors = self._branch_seed_programs(adapter, seed_pool, train_obs)
        all_programs.extend(branched)
        all_errors.extend(branch_errors)
        if layer_programs or self_modules or seeded or typed_priors or composed or branched:
            ranked_seed = self._rank(all_programs, hold_obs, adapter)
            if (
                ranked_seed
                and ranked_seed[0][1].loss_mean <= _ZERO_LOSS_EPS
                and not self._nova_enabled()
            ):
                best_seed, _seed_score = ranked_seed[0]
                if not (
                    _uninformative_angle_numeric_task(adapter, train_obs)
                    and _program_is_degenerate_constant(best_seed, train_obs)
                ):
                    return self._finalize_result(
                        adapter=adapter,
                        observations=observations,
                        programs=all_programs,
                        proposals_raw=self._with_darwin_trace_proposals(adapter, train_obs, all_proposals),
                        errors=all_errors,
                        t0=t0,
                        rounds=0,
                        n_proposed=0,
                    )
            if ranked_seed:
                best_seed, seed_score = ranked_seed[0]
                failure_context = (
                    f"\nTRUSTED SEED/COMPOSITION PROGRAM '{best_seed.name}' was "
                    f"loaded or composed from the self-written module library but "
                    f"is not perfect on this task "
                    f"(held-out loss={seed_score.loss_mean:.3f}). Mutate/repair this "
                    "program first, then propose genuinely different alternatives."
                    f"\nTRUSTED SEED CODE:\n```python\n{best_seed.code[:1600]}\n```"
                    + self._failure_examples(best_seed, hold_obs, adapter)
                )
                residual_programs, residual_proposals, residual_errors, _ = (
                    self._propose_residual_operator_layers(
                        adapter,
                        train_obs,
                        seed_pool,
                        failure_context=failure_context,
                    )
                )
                residual_programs = [
                    adapter.calibrate(p, train_obs) for p in residual_programs
                ]
                all_programs.extend(residual_programs)
                all_proposals.extend(residual_proposals)
                all_errors.extend(residual_errors)
                if residual_programs:
                    ranked_residual = self._rank(all_programs, hold_obs, adapter)
                    if ranked_residual and ranked_residual[0][1].loss_mean <= _ZERO_LOSS_EPS:
                        return self._finalize_result(
                            adapter=adapter,
                            observations=observations,
                            programs=all_programs,
                            proposals_raw=self._with_darwin_trace_proposals(adapter, train_obs, all_proposals),
                            errors=all_errors,
                            t0=t0,
                            rounds=0,
                            n_proposed=len(all_proposals),
                        )

        nova_programs, nova_errors, nova_proposals = self._nova_programs(
            adapter,
            train_obs,
        )
        all_programs.extend(nova_programs)
        all_errors.extend(nova_errors)
        all_proposals.extend(nova_proposals)
        if nova_programs:
            ranked_nova = self._rank(all_programs, hold_obs, adapter)
            if ranked_nova and ranked_nova[0][1].loss_mean <= _ZERO_LOSS_EPS:
                return self._finalize_result(
                    adapter=adapter,
                    observations=observations,
                    programs=all_programs,
                    proposals_raw=self._with_darwin_trace_proposals(adapter, train_obs, all_proposals),
                    errors=all_errors,
                    t0=t0,
                    rounds=0,
                    n_proposed=len(all_proposals),
                )
            if ranked_nova:
                best_nova, nova_score = ranked_nova[0]
                failure_context += (
                    f"\nNOVA local-oracle prior produced best candidate "
                    f"'{best_nova.name}' with held-out loss={nova_score.loss_mean:.3f}. "
                    "Use this as measured prior evidence; repair only the residuals "
                    "that remain."
                    f"\nNOVA BEST CODE:\n```python\n{best_nova.code[:1600]}\n```"
                    + self._failure_examples(best_nova, hold_obs, adapter)
                )

        syndrome_programs, syndrome_errors, syndrome_proposals = self._syndrome_programs(
            adapter,
            train_obs,
            all_programs,
        )
        all_programs.extend(syndrome_programs)
        all_errors.extend(syndrome_errors)
        all_proposals.extend(syndrome_proposals)
        if syndrome_programs:
            ranked_syndrome = self._rank(all_programs, hold_obs, adapter)
            if ranked_syndrome and ranked_syndrome[0][1].loss_mean <= _ZERO_LOSS_EPS:
                return self._finalize_result(
                    adapter=adapter,
                    observations=observations,
                    programs=all_programs,
                    proposals_raw=self._with_darwin_trace_proposals(adapter, train_obs, all_proposals),
                    errors=all_errors,
                    t0=t0,
                    rounds=0,
                    n_proposed=len(all_proposals),
                )
            if ranked_syndrome:
                best_syndrome, syndrome_score = ranked_syndrome[0]
                failure_context += (
                    f"\nSYNDROME DECODER produced best candidate "
                    f"'{best_syndrome.name}' with held-out loss={syndrome_score.loss_mean:.3f}. "
                    "Treat this as a decoded missing-operator repair, not as a "
                    "free-form model guess."
                    f"\nSYNDROME BEST CODE:\n```python\n{best_syndrome.code[:1600]}\n```"
                    + self._failure_examples(best_syndrome, hold_obs, adapter)
                )

        darwin_programs, darwin_errors, darwin_proposals = self._darwin_programs(
            adapter,
            train_obs,
        )
        all_programs.extend(darwin_programs)
        all_errors.extend(darwin_errors)
        all_proposals.extend(darwin_proposals)
        if darwin_programs:
            ranked_darwin = self._rank(all_programs, hold_obs, adapter)
            if ranked_darwin and ranked_darwin[0][1].loss_mean <= _ZERO_LOSS_EPS:
                return self._finalize_result(
                    adapter=adapter,
                    observations=observations,
                    programs=all_programs,
                    proposals_raw=self._with_darwin_trace_proposals(adapter, train_obs, all_proposals),
                    errors=all_errors,
                    t0=t0,
                    rounds=0,
                    n_proposed=len(all_proposals),
                )
            if ranked_darwin:
                best_darwin, darwin_score = ranked_darwin[0]
                failure_context += (
                    f"\nDARWIN theory population produced best phenotype "
                    f"'{best_darwin.name}' with held-out loss={darwin_score.loss_mean:.3f}. "
                    "Treat this as selected formal theory genome, not an LLM proposal."
                    f"\nDARWIN BEST CODE:\n```python\n{best_darwin.code[:1600]}\n```"
                    + self._failure_examples(best_darwin, hold_obs, adapter)
                )

        for rnd in range(self.max_rounds):
            rounds = rnd + 1
            generated_layer_programs, layer_proposals, layer_errors, proposed_layer_outputs = (
                self._propose_self_layers(adapter, train_obs, failure_context)
            )
            generated_layer_programs = [
                adapter.calibrate(p, train_obs) for p in generated_layer_programs
            ]
            all_programs.extend(generated_layer_programs)
            generated_operator_programs, generated_operator_errors = (
                self._programs_from_layer_operators(
                    adapter,
                    proposed_layer_outputs,
                    all_programs,
                    train_obs,
                    failure_context=failure_context,
                )
            )
            all_programs.extend(generated_operator_programs)
            all_proposals.extend(
                {"kind": "self_layer", **p} for p in layer_proposals if isinstance(p, dict)
            )
            all_errors.extend(layer_errors)
            all_errors.extend(generated_operator_errors)
            if proposed_layer_outputs:
                all_errors.append(
                    "proposed self layers emitted "
                    f"{sum(len(list(o.get('program_sources') or [])) for o in proposed_layer_outputs)} "
                    "program sources and "
                    f"{sum(len(list(o.get('operator_sources') or [])) for o in proposed_layer_outputs)} "
                    "operator sources"
                )
            if all_programs:
                ranked_after_layers = self._rank(all_programs, hold_obs, adapter)
                if ranked_after_layers and ranked_after_layers[0][1].loss_mean <= _ZERO_LOSS_EPS:
                    break

            residual_programs, residual_proposals, residual_errors, _ = (
                self._propose_residual_operator_layers(
                    adapter,
                    train_obs,
                    all_programs,
                    failure_context=failure_context,
                )
            )
            residual_programs = [
                adapter.calibrate(p, train_obs) for p in residual_programs
            ]
            all_programs.extend(residual_programs)
            all_proposals.extend(residual_proposals)
            all_errors.extend(residual_errors)
            if residual_programs:
                ranked_after_residual = self._rank(all_programs, hold_obs, adapter)
                if (
                    ranked_after_residual
                    and ranked_after_residual[0][1].loss_mean <= _ZERO_LOSS_EPS
                ):
                    break

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

            if best_loss <= _ZERO_LOSS_EPS:
                break  # perfect fit, stop early

            # Build failure context for next round from the best imperfect program
            best_prog, best_score = ranked[0]
            failure_context = (
                f"\nPREVIOUS ROUND best program '{best_prog.name}' achieved "
                f"loss={best_score.loss_mean:.3f} (not perfect). Propose NEW mechanisms "
                f"that differ structurally — consider interactions, conditionals on "
                f"context variables, history, and compositions you have not tried."
                f"\nPREVIOUS BEST PROGRAM CODE:\n```python\n{best_prog.code[:1600]}\n```"
                + self._failure_examples(best_prog, hold_obs, adapter)
            )

        return self._finalize_result(
            adapter=adapter,
            observations=observations,
            programs=all_programs,
            proposals_raw=self._with_darwin_trace_proposals(adapter, train_obs, all_proposals),
            errors=all_errors,
            t0=t0,
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


def _parse_layers(raw: str) -> list[dict]:
    if not raw:
        return []
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).lstrip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            for key in ("layers", "self_layers", "architecture_layers"):
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
                for key in ("layers", "self_layers", "architecture_layers"):
                    if isinstance(obj.get(key), list):
                        return obj[key]
        except Exception:
            pass
    return []


def _parse_operator_plans(raw: str) -> list[dict]:
    """Parse the intentionally small JSON protocol for typed plan selection."""

    if not raw:
        return []
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).lstrip()
    try:
        value = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return []
        try:
            value = json.loads(match.group())
        except Exception:
            return []
    if isinstance(value, dict) and isinstance(value.get("plans"), list):
        return [item for item in value["plans"] if isinstance(item, dict)]
    return []


def _safe_name(value: str) -> str:
    name = re.sub(r"[^0-9a-zA-Z_]+", "_", value).strip("_").lower()
    return name or "predicate"
