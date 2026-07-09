"""Self-induced benchmark theory compiler.

This layer sits above metric-specific hypothesis compilation.  It does not try
to solve a benchmark directly.  Instead it induces the *interface theory* MARS
needs before solving:

  task object -> allowed actions -> metric contract -> verifier graph
  -> residual classes -> reusable skill schemas -> compression objective

The implementation is intentionally deterministic at v0.  It uses lexical and
structural evidence from benchmark READMEs, paper notes, evaluator docs, and
local run summaries.  A stronger future version can replace the lexical
extractors with program analysis or LLM-assisted spec extraction while keeping
the same theory object.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import re
from typing import Any, Mapping

from .metric_compiler import tokenize


@dataclass(frozen=True)
class ActionSpec:
    name: str
    description: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    verifier_signal: str


@dataclass(frozen=True)
class MetricSpec:
    name: str
    score_range: str
    target_artifact: str
    evaluator: str
    facets: tuple[str, ...] = ()


@dataclass(frozen=True)
class CheckSpec:
    name: str
    checks: str
    failure_signal: str
    residual_class: str


@dataclass(frozen=True)
class ResidualSpec:
    name: str
    trigger: str
    observable_fields: tuple[str, ...]
    repair_operator: str
    priority: float = 1.0


@dataclass(frozen=True)
class TheorySkillSchema:
    name: str
    detector: str
    generator: str
    verifier: str
    repair: str
    compression_key: tuple[str, ...]


@dataclass(frozen=True)
class BenchmarkTheory:
    benchmark_name: str
    theory_id: str
    task_object: str
    observable_inputs: tuple[str, ...]
    target_outputs: tuple[str, ...]
    actions: tuple[ActionSpec, ...]
    metrics: tuple[MetricSpec, ...]
    checks: tuple[CheckSpec, ...]
    residuals: tuple[ResidualSpec, ...]
    skill_schemas: tuple[TheorySkillSchema, ...]
    compression_objective: str
    evidence: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    confidence: float = 0.0
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


INPUT_LEXICON: dict[str, set[str]] = {
    "datasets": {"csv", "data", "dataset", "datasets", "table", "tables"},
    "metadata": {"metadata", "schema", "description", "descriptions"},
    "files": {"file", "files", "program", "python", "workspace"},
    "environment": {"environment", "env", "interactive", "tool", "tools"},
    "trajectories": {"trajectory", "trajectories", "trace", "traces", "sequence", "sequences"},
    "observations": {"experiment", "experiments", "observation", "observations", "measurement", "measurements"},
    "natural_language_goal": {"goal", "instruction", "query", "question", "task"},
}

OUTPUT_LEXICON: dict[str, set[str]] = {
    "hypothesis": {"hypothesis", "hypotheses", "claim", "discovery"},
    "program": {"code", "program", "script", "python"},
    "law": {"formula", "law", "symbolic", "equation"},
    "policy_or_report": {"answer", "final", "policy", "report", "result", "submission"},
    "transition_rule": {"rule", "transition", "transformation", "simulator"},
}

ACTION_TEMPLATES: tuple[tuple[str, set[str], ActionSpec], ...] = (
    (
        "inspect_data",
        {"data", "dataset", "metadata", "schema", "table", "csv"},
        ActionSpec(
            name="inspect_data",
            description="Inspect available data, schema, metadata, and observed values before proposing an answer.",
            inputs=("datasets", "metadata", "goal"),
            outputs=("data_profile", "candidate_fields"),
            verifier_signal="field coverage and schema consistency",
        ),
    ),
    (
        "execute_program",
        {"code", "program", "python", "execute", "execution", "tool"},
        ActionSpec(
            name="execute_program",
            description="Generate or run code/tool calls and observe execution results.",
            inputs=("files", "tool_signatures", "candidate_program"),
            outputs=("execution_trace", "artifact"),
            verifier_signal="runtime success, saved output, and test result",
        ),
    ),
    (
        "run_experiment",
        {"experiment", "interactive", "observation", "measurement"},
        ActionSpec(
            name="run_experiment",
            description="Choose interventions or measurements that reveal hidden structure.",
            inputs=("state", "action_choice", "budget"),
            outputs=("observation", "measurement"),
            verifier_signal="new evidence reduces candidate uncertainty",
        ),
    ),
    (
        "rollout_candidate",
        {"horizon", "sequence", "trajectory", "transition", "rollout"},
        ActionSpec(
            name="rollout_candidate",
            description="Run a candidate transition rule over future steps and compare with held-out traces.",
            inputs=("state", "candidate_rule", "horizon"),
            outputs=("predicted_trace", "rollout_error"),
            verifier_signal="first failing step and rollout match",
        ),
    ),
    (
        "fit_and_test_law",
        {"formula", "law", "numeric", "prediction", "rmsle", "symbolic"},
        ActionSpec(
            name="fit_and_test_law",
            description="Fit compact law families and test them on held-out numeric cases.",
            inputs=("observations", "candidate_formula_family"),
            outputs=("fitted_law", "prediction_error"),
            verifier_signal="symbolic equivalence and numeric prediction error",
        ),
    ),
    (
        "score_hypothesis",
        {"hypothesis", "hms", "relation", "variables", "context"},
        ActionSpec(
            name="score_hypothesis",
            description="Compile an answer into evaluator-facing hypothesis slots and score slot agreement.",
            inputs=("goal", "evidence", "candidate_hypothesis"),
            outputs=("slot_scores", "semantic_residuals"),
            verifier_signal="context, variable, relation, evidence, and output slot agreement",
        ),
    ),
)

METRIC_PATTERNS: tuple[tuple[str, str, str, str, tuple[str, ...]], ...] = (
    ("HMS", r"\bHMS\b|hypothesis matching", "0-100", "natural-language hypothesis", "LLM semantic sub-hypothesis evaluator", ("context", "variables", "relation")),
    ("SA", r"symbolic accuracy|\bSA\b", "0-100%", "symbolic law", "symbolic equivalence judge", ("law_structure", "constants", "equivalence")),
    ("RMSLE", r"\bRMSLE\b", "0-inf lower is better", "numeric law predictions", "held-out numeric evaluator", ("prediction_error",)),
    ("SR", r"success rate|\bSR\b", "0-100%", "executed scientific workflow", "task-specific success grader", ("answer_correctness",)),
    ("VER", r"valid execution|\bVER\b", "0-100%", "executable program", "runtime/output validator", ("execution_validity",)),
    ("Average score", r"average score|final_score|score@k|task score", "benchmark-defined", "final report or policy", "point-wise judge or environment scorer", ("subscores", "judge_feedback")),
    ("Exact fit", r"exact program|exact fit|exact rule", "count or rate", "program rule", "deterministic rule matcher", ("rule_match",)),
)

RESIDUAL_LIBRARY: tuple[ResidualSpec, ...] = (
    ResidualSpec(
        name="metric_slot_mismatch",
        trigger="candidate answer misses context, variable, relation, evidence, or output slots",
        observable_fields=("slot_scores", "gold_subhypothesis", "predicted_hypothesis"),
        repair_operator="metric_aligned_hypothesis_compiler",
        priority=1.25,
    ),
    ResidualSpec(
        name="data_binding_error",
        trigger="candidate references unavailable files, columns, variables, or observed values",
        observable_fields=("schema", "metadata", "execution_trace", "candidate_fields"),
        repair_operator="schema_and_value_linker",
        priority=1.15,
    ),
    ResidualSpec(
        name="execution_error",
        trigger="candidate code or tool call fails before producing the scored artifact",
        observable_fields=("stderr", "traceback", "missing_output", "tool_signature"),
        repair_operator="execution_precondition_repair",
        priority=1.05,
    ),
    ResidualSpec(
        name="numeric_law_error",
        trigger="candidate formula has high held-out prediction error or wrong symbolic structure",
        observable_fields=("observations", "candidate_formula", "rmsle", "symbolic_judge"),
        repair_operator="formula_family_search",
        priority=1.1,
    ),
    ResidualSpec(
        name="rollout_drift",
        trigger="candidate transition rule fits early observations but fails over horizon",
        observable_fields=("trace", "candidate_rule", "first_failing_step", "rollout_error"),
        repair_operator="hidden_state_or_branch_repair",
        priority=1.1,
    ),
    ResidualSpec(
        name="scientific_interpretation_error",
        trigger="workflow executes but final scientific answer is unsupported or wrong",
        observable_fields=("artifact", "rubric", "analysis_trace", "answer"),
        repair_operator="result_interpretation_verifier",
        priority=1.0,
    ),
)

SKILL_LIBRARY: dict[str, TheorySkillSchema] = {
    "metric_slot_mismatch": TheorySkillSchema(
        name="metric_contract_skill",
        detector="find missing evaluator-facing slots in candidate artifacts",
        generator="emit typed hypothesis skeletons before free-form output",
        verifier="score context/variable/relation/evidence/output agreement",
        repair="rewrite candidate through the missing slots",
        compression_key=("metric", "slots", "semantic_residual"),
    ),
    "data_binding_error": TheorySkillSchema(
        name="schema_grounding_skill",
        detector="compare candidate references against observed schema and values",
        generator="bind variables/files/categories from local artifacts",
        verifier="all candidate references resolve to available data",
        repair="replace unsupported references with grounded alternatives",
        compression_key=("schema", "values", "entity_linking"),
    ),
    "execution_error": TheorySkillSchema(
        name="execution_preflight_skill",
        detector="classify runtime, import, path, output, and tool-signature errors",
        generator="insert precondition checks before final execution",
        verifier="program runs and produces required files",
        repair="patch code or tool call using typed execution residual",
        compression_key=("execution", "precondition", "repair"),
    ),
    "numeric_law_error": TheorySkillSchema(
        name="law_search_skill",
        detector="locate formula families with high train/held-out mismatch",
        generator="fit compact symbolic families under held-out verification",
        verifier="numeric error and symbolic equivalence pass",
        repair="add missing variable, regime, exponent, or family",
        compression_key=("law", "fit", "heldout"),
    ),
    "rollout_drift": TheorySkillSchema(
        name="transition_simulator_skill",
        detector="find first rollout step where predicted state diverges",
        generator="induce executable transition simulators from traces",
        verifier="candidate rule survives full horizon",
        repair="add hidden state, branch, memory, or delayed effect",
        compression_key=("transition", "rollout", "hidden_state"),
    ),
    "scientific_interpretation_error": TheorySkillSchema(
        name="result_interpretation_skill",
        detector="separate valid execution from unsupported scientific conclusion",
        generator="link artifact statistics to final answer claims",
        verifier="rubric and result consistency checks pass",
        repair="revise conclusion or rerun missing analysis",
        compression_key=("science", "artifact", "interpretation"),
    ),
}


def _join_sources(sources: Mapping[str, str]) -> str:
    return "\n\n".join(f"# {name}\n{text}" for name, text in sources.items())


def _theory_tokens(text: str) -> set[str]:
    out: set[str] = set()
    for token in tokenize(text):
        cleaned = token.strip(".,;:()[]{}<>!?\"'")
        if cleaned:
            out.add(cleaned)
    return out


def _evidence(tokens: set[str], lexicon: Mapping[str, set[str]]) -> dict[str, tuple[str, ...]]:
    found: dict[str, tuple[str, ...]] = {}
    for name, words in lexicon.items():
        hits = tuple(sorted(tokens & words))
        if hits:
            found[name] = hits
    return found


def _infer_task_object(inputs: tuple[str, ...], outputs: tuple[str, ...]) -> str:
    if "law" in outputs:
        return "interactive scientific law induction task"
    if "transition_rule" in outputs:
        return "long-horizon transition program induction task"
    if "program" in outputs:
        return "scientific workflow programming task"
    if "hypothesis" in outputs:
        return "data-grounded hypothesis generation task"
    if "environment" in inputs:
        return "interactive benchmark environment task"
    return "open-ended benchmark task with inferred verifier"


def _infer_metrics(text: str) -> tuple[MetricSpec, ...]:
    metrics: list[MetricSpec] = []
    for name, pattern, score_range, artifact, evaluator, facets in METRIC_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            metrics.append(
                MetricSpec(
                    name=name,
                    score_range=score_range,
                    target_artifact=artifact,
                    evaluator=evaluator,
                    facets=facets,
                )
            )
    if not metrics:
        metrics.append(
            MetricSpec(
                name="task_score",
                score_range="benchmark-defined",
                target_artifact="final artifact",
                evaluator="benchmark evaluator",
                facets=("artifact_validity", "task_success"),
            )
        )
    return tuple(metrics)


def _infer_actions(tokens: set[str]) -> tuple[ActionSpec, ...]:
    actions: list[ActionSpec] = []
    for _, words, spec in ACTION_TEMPLATES:
        if tokens & words:
            actions.append(spec)
    if not actions:
        actions.append(
            ActionSpec(
                name="propose_and_score",
                description="Propose candidate artifacts and ask the benchmark verifier for residuals.",
                inputs=("task", "candidate_artifact"),
                outputs=("score", "residual"),
                verifier_signal="benchmark score or failure reason",
            )
        )
    return tuple(actions)


def _infer_residuals(
    tokens: set[str],
    metrics: tuple[MetricSpec, ...],
    actions: tuple[ActionSpec, ...],
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
) -> tuple[ResidualSpec, ...]:
    wanted: set[str] = set()
    metric_names = {m.name for m in metrics}
    action_names = {a.name for a in actions}
    if metric_names & {"HMS"} or "hypothesis" in outputs:
        wanted.add("metric_slot_mismatch")
    if {"datasets", "metadata", "files"} & set(inputs):
        wanted.add("data_binding_error")
    if metric_names & {"VER", "SR"} or ("program" in outputs and action_names & {"execute_program"}):
        wanted.add("execution_error")
    if metric_names & {"SA", "RMSLE"} or "law" in outputs:
        wanted.add("numeric_law_error")
    if action_names & {"rollout_candidate"} or "transition_rule" in outputs:
        wanted.add("rollout_drift")
    if metric_names & {"SR"} and ("program" in outputs or "files" in inputs):
        wanted.add("scientific_interpretation_error")
    if not wanted:
        wanted.add("metric_slot_mismatch")
        wanted.add("data_binding_error")
    residuals = [r for r in RESIDUAL_LIBRARY if r.name in wanted]
    return tuple(sorted(residuals, key=lambda r: (-r.priority, r.name)))


def _checks_for_residuals(metrics: tuple[MetricSpec, ...], residuals: tuple[ResidualSpec, ...]) -> tuple[CheckSpec, ...]:
    metric_text = ", ".join(m.name for m in metrics)
    return tuple(
        CheckSpec(
            name=f"check_{residual.name}",
            checks=f"{metric_text} artifact against residual class `{residual.name}`",
            failure_signal=residual.trigger,
            residual_class=residual.name,
        )
        for residual in residuals
    )


def _skill_schemas(residuals: tuple[ResidualSpec, ...]) -> tuple[TheorySkillSchema, ...]:
    return tuple(SKILL_LIBRARY[r.name] for r in residuals if r.name in SKILL_LIBRARY)


def _select_actions(actions: tuple[ActionSpec, ...], names: set[str]) -> tuple[ActionSpec, ...]:
    selected = tuple(a for a in actions if a.name in names)
    if selected:
        return selected
    by_name = {a.name: a for _, _, a in ACTION_TEMPLATES}
    return tuple(by_name[n] for n in names if n in by_name)


def _prune_by_metric_theory(
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
    actions: tuple[ActionSpec, ...],
    metrics: tuple[MetricSpec, ...],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[ActionSpec, ...]]:
    """Use metric identity as the primary contract for interface pruning."""

    names = {m.name for m in metrics}
    if "HMS" in names:
        keep_inputs = tuple(x for x in inputs if x in {"datasets", "metadata", "natural_language_goal", "files"})
        keep_outputs = ("hypothesis",)
        keep_actions = _select_actions(actions, {"inspect_data", "score_hypothesis"})
        return keep_inputs or ("datasets", "metadata", "natural_language_goal"), keep_outputs, keep_actions
    if names & {"SA", "RMSLE"}:
        keep_inputs = tuple(x for x in inputs if x in {"environment", "observations", "natural_language_goal"})
        keep_outputs = ("law",)
        keep_actions = _select_actions(actions, {"run_experiment", "fit_and_test_law"})
        return keep_inputs or ("observations", "natural_language_goal"), keep_outputs, keep_actions
    if names & {"SR", "VER"}:
        keep_inputs = tuple(x for x in inputs if x in {"files", "datasets", "metadata", "natural_language_goal", "environment"})
        keep_outputs = ("program",)
        keep_actions = _select_actions(actions, {"inspect_data", "execute_program"})
        return keep_inputs or ("files", "natural_language_goal"), keep_outputs, keep_actions
    if names & {"Average score", "Exact fit"}:
        keep_inputs = tuple(x for x in inputs if x in {"environment", "trajectories", "observations", "natural_language_goal"})
        keep_outputs = ("transition_rule", "policy_or_report")
        keep_actions = _select_actions(actions, {"run_experiment", "rollout_candidate"})
        return keep_inputs or ("environment", "trajectories"), keep_outputs, keep_actions
    return inputs, outputs, actions


def _theory_id(payload: Mapping[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(data.encode("utf-8")).hexdigest()[:12]


def _confidence(
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
    actions: tuple[ActionSpec, ...],
    metrics: tuple[MetricSpec, ...],
    residuals: tuple[ResidualSpec, ...],
) -> float:
    pieces = [
        min(1.0, len(inputs) / 4.0),
        min(1.0, len(outputs) / 3.0),
        min(1.0, len(actions) / 3.0),
        1.0 if metrics and metrics[0].name != "task_score" else 0.45,
        min(1.0, len(residuals) / 4.0),
    ]
    return round(sum(pieces) / len(pieces), 3)


def compile_benchmark_theory(
    benchmark_name: str,
    sources: Mapping[str, str],
    *,
    prior_results: Mapping[str, Any] | None = None,
) -> BenchmarkTheory:
    """Induce a benchmark interface theory from textual/code evidence.

    `sources` should contain whatever is locally available: paper notes,
    README excerpts, evaluator help text, and run summaries.  The compiler is
    intentionally not given benchmark-specific Python classes.
    """

    source_text = _join_sources(sources)
    if prior_results:
        source_text += "\n\n# prior_results\n" + json.dumps(prior_results, ensure_ascii=False, default=str)
    tokens = _theory_tokens(source_text)
    input_evidence = _evidence(tokens, INPUT_LEXICON)
    output_evidence = _evidence(tokens, OUTPUT_LEXICON)
    inputs = tuple(sorted(input_evidence)) or ("task",)
    outputs = tuple(sorted(output_evidence)) or ("final_artifact",)
    metrics = _infer_metrics(source_text)
    actions = _infer_actions(tokens)
    inputs, outputs, actions = _prune_by_metric_theory(inputs, outputs, actions, metrics)
    residuals = _infer_residuals(tokens, metrics, actions, inputs, outputs)
    checks = _checks_for_residuals(metrics, residuals)
    skills = _skill_schemas(residuals)
    task_object = _infer_task_object(inputs, outputs)
    notes = []
    if any(m.name == "task_score" for m in metrics):
        notes.append("Metric was not explicit in sources; use benchmark evaluator discovery before final runs.")
    if prior_results:
        notes.append("Prior local run summaries were included as residual evidence, not as a solver shortcut.")
    payload = {
        "benchmark_name": benchmark_name,
        "task_object": task_object,
        "inputs": inputs,
        "outputs": outputs,
        "metrics": [m.name for m in metrics],
        "actions": [a.name for a in actions],
        "residuals": [r.name for r in residuals],
        "skills": [s.name for s in skills],
    }
    return BenchmarkTheory(
        benchmark_name=benchmark_name,
        theory_id=_theory_id(payload),
        task_object=task_object,
        observable_inputs=inputs,
        target_outputs=outputs,
        actions=actions,
        metrics=metrics,
        checks=checks,
        residuals=residuals,
        skill_schemas=skills,
        compression_objective=(
            "argmin_T L(interface_T) + L(action_grammar_T) + "
            "L(verifier_graph_T) + L(residuals | T); promote skills only when "
            "program_bits(skill) + residual_bits_after < residual_bits_before"
        ),
        evidence={
            "inputs": tuple(f"{k}:{','.join(v)}" for k, v in sorted(input_evidence.items())),
            "outputs": tuple(f"{k}:{','.join(v)}" for k, v in sorted(output_evidence.items())),
        },
        confidence=_confidence(inputs, outputs, actions, metrics, residuals),
        notes=tuple(notes),
    )
