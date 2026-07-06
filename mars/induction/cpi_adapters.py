"""Thin CPI adapters for the 4 competition benchmarks.

Each adapter binds the universal engine to one environment. It contains ONLY:
  - the observable interface schema (types, variables — NOT answers)
  - how to collect observations
  - how to execute a candidate program
  - how to measure loss against an observation
  - how to render winning programs into the benchmark's output

No adapter prints a domain answer. Every hypothesis is discovered by the engine
through LLM proposal + sandbox + refutation scoring on held-out observations.
"""

from __future__ import annotations

import json
import re
from typing import Any

from mars.agents.base import call_llm, make_openai_client
from mars.induction.universal_cpi import (
    CPIAdapter,
    HypothesisProgram,
    Observation,
    ProgramScore,
)
from mars.induction.universal_hypothesis_kernel import KernelTask, UniversalHypothesisKernel
from mars.skills.metric_compiler import (
    compile_hypothesis_workbench,
    infer_temporal_event_hypothesis,
)
from mars.skills.slot_contract import infer_slot_contract_hypothesis
from mars.skills.universal_slot_compiler import infer_universal_slot_hypothesis
from mars.skills.problem_frame_inducer import infer_problem_frame_hypothesis
from mars.skills.contrastive_world_inducer import infer_contrastive_world_hypothesis
from mars.skills.answer_plan_inducer import infer_answer_plan_hypothesis
from mars.skills.contract_baselines import generate_dataframe_analyzer_baselines
from mars.skills.answer_contract import compile_answer_contract, normalize_answer_report
from mars.skills.evidence_contract_compiler import infer_evidence_contract_hypothesis


# ===========================================================================
# 1. UltraHorizon Sequence — hidden string transformation rules
# ===========================================================================

class UHSeqAdapter(CPIAdapter):
    name = "uh_seq"

    def __init__(
        self,
        env: Any,
        rule_slot: int,
        steps: int = 5,
        induction_library: list[dict[str, Any]] | None = None,
    ):
        self.env = env
        self.rule_slot = rule_slot   # which of 5 rules we induce (1..5)
        self.steps = steps
        self.induction_library = induction_library or []
        self._observations: list[Observation] | None = None
        self._raw_results: list[dict] = []

    def interface_description(self) -> str:
        return (
            "Environment: hidden string-transformation rule.\n"
            "A rule maps the current string + context to the next string.\n"
            "context['main'], context['vice']: 5-letter uppercase inputs (A-E).\n"
            "context['step_number']: int. context['previous_main']: prior main string.\n"
            "Output: an uppercase string. Char position: ord(c)-ord('A'); "
            "char at i: chr(ord('A') + i % 26)."
        )

    def signature_hint(self) -> str:
        return "def rule(current: str, context: dict) -> str"

    def induction_library_hint(self) -> str:
        if not self.induction_library:
            return ""
        lines = [
            "\nDISCOVERED PARTIAL PROGRAM LIBRARY (scored on earlier slots; reuse ideas only if they reduce residuals):"
        ]
        for item in self.induction_library[-6:]:
            lines.append(
                f"- slot={item.get('slot')} name={item.get('name')} "
                f"loss={item.get('loss_mean')} exact={item.get('exact_rate')} "
                f"description={str(item.get('description', ''))[:180]}\n"
                f"```python\n{str(item.get('code', ''))[:900]}\n```"
            )
        return "\n".join(lines)

    def seed_program_sources(self) -> list[Any]:
        seeds: list[Any] = []
        for item in self.induction_library:
            code = str(item.get("code", "") or "").strip()
            if not code:
                continue
            seeds.append(
                {
                    "name": item.get("name", "trusted_seed"),
                    "description": item.get("description", "trusted seed program"),
                    "complexity": 1.2,
                    "code": code,
                }
            )
        return seeds

    def compose_seed_programs(self) -> bool:
        return True

    def branch_seed_programs(self) -> bool:
        return True

    def branch_predicate_sources(self, observations) -> list[dict[str, str]]:
        context_keys = set()
        for obs in observations:
            context_keys.update(obs.context.keys())

        predicates: list[dict[str, str]] = []
        if "step_number" in context_keys:
            predicates.extend([
                {
                    "name": "step_even",
                    "expr": "int(context.get('step_number', 0)) % 2 == 0",
                },
                {
                    "name": "step_odd",
                    "expr": "int(context.get('step_number', 0)) % 2 == 1",
                },
                {
                    "name": "step_prime",
                    "expr": "_is_prime(int(context.get('step_number', 0)))",
                    "helper": (
                        "def _is_prime(n: int) -> bool:\n"
                        "    if n < 2:\n"
                        "        return False\n"
                        "    d = 2\n"
                        "    while d * d <= n:\n"
                        "        if n % d == 0:\n"
                        "            return False\n"
                        "        d += 1\n"
                        "    return True\n"
                    ),
                },
                {
                    "name": "step_not_prime",
                    "expr": "not _is_prime(int(context.get('step_number', 0)))",
                    "helper": (
                        "def _is_prime(n: int) -> bool:\n"
                        "    if n < 2:\n"
                        "        return False\n"
                        "    d = 2\n"
                        "    while d * d <= n:\n"
                        "        if n % d == 0:\n"
                        "            return False\n"
                        "        d += 1\n"
                        "    return True\n"
                    ),
                },
            ])

        predicates.extend([
            {
                "name": "empty_current",
                "expr": "len(str(current)) == 0",
            },
            {
                "name": "current_length_even",
                "expr": "len(str(current)) % 2 == 0",
            },
            {
                "name": "current_length_odd",
                "expr": "len(str(current)) % 2 == 1",
            },
        ])
        return predicates

    def set_observations(self, raw_results: list[dict]) -> None:
        """Inject already-collected env results (shared across 5 rule slots)."""
        self._raw_results = raw_results

    def collect_observations(self) -> list[Observation]:
        if self._observations is not None:
            return self._observations
        obs: list[Observation] = []
        prev_main = ""
        for r in self._raw_results:
            transforms = r.get("transformations", [])
            outs = [str(t.get("sequence", "")) for t in transforms[1:6]]
            if len(outs) != 5:
                continue
            current = "" if self.rule_slot == 1 else outs[self.rule_slot - 2]
            target = outs[self.rule_slot - 1]
            obs.append(Observation(
                inputs=current,
                target=target,
                context={
                    "main": str(r.get("main_input", "")),
                    "vice": str(r.get("vice_input", "")),
                    "step_number": int(r.get("step_number", 0)),
                    "previous_main": prev_main,
                },
            ))
            prev_main = str(r.get("main_input", ""))
        self._observations = obs
        return obs

    def execute(self, program: HypothesisProgram, obs: Observation) -> Any:
        return program.fn(obs.inputs, obs.context)

    def loss(self, prediction: Any, obs: Observation) -> float:
        if prediction == obs.target:
            return 0.0
        p = str(prediction)
        t = str(obs.target)
        denom = max(1, len(p), len(t))
        return _edit_distance(p, t) / denom

    def render_report(self, winners, observations) -> str:
        if not winners:
            return ""
        best, score = winners[0]
        return best.description

    def auto_signals(self, observations) -> str:
        if not observations:
            return ""
        in_lens = [len(str(o.inputs)) for o in observations]
        out_lens = [len(str(o.target)) for o in observations]
        steps = [o.context.get("step_number", 0) for o in observations]
        hints = ["\nDATA-DERIVED SIGNALS (facts, not answers):"]
        if all(ol == 2 * il for il, ol in zip(in_lens, out_lens) if il > 0):
            hints.append("- output length is exactly 2x input length")
        elif all(ol == il + s for il, ol, s in zip(in_lens, out_lens, steps)):
            hints.append("- output length = input length + step_number")
        elif all(ol == il for il, ol in zip(in_lens, out_lens)):
            hints.append("- output length equals input length")
        return "\n".join(hints) if len(hints) > 1 else ""


# ===========================================================================
# 2. NewtonBench — hidden physical law (symbolic regression)
# ===========================================================================

class NewtonAdapter(CPIAdapter):
    name = "newtonbench"

    def __init__(self, data_points: list[dict], var_names: list[str], target_name: str):
        # data_points: list of {var: value, ..., target_name: value}
        self.data_points = data_points
        self.var_names = var_names
        self.target_name = target_name
        self._observations: list[Observation] | None = None

    def interface_description(self) -> str:
        return (
            "Environment: hidden physical law producing a scalar output.\n"
            f"Input variables: {self.var_names}.\n"
            f"Output: scalar '{self.target_name}'.\n"
            "Propose a closed-form law. Use math (math.sqrt, math.pi, etc.).\n"
            "Inputs arrive as a dict {var_name: float}."
        )

    def signature_hint(self) -> str:
        return f"def law(inputs: dict) -> float  # inputs has keys {self.var_names}"

    def collect_observations(self) -> list[Observation]:
        if self._observations is not None:
            return self._observations
        obs = []
        for dp in self.data_points:
            inp = {v: float(dp[v]) for v in self.var_names if v in dp}
            tgt = float(dp[self.target_name])
            obs.append(Observation(inputs=inp, target=tgt, context={}))
        self._observations = obs
        return obs

    def execute(self, program: HypothesisProgram, obs: Observation) -> Any:
        k = getattr(program, "_const", 1.0)
        return k * program.fn(obs.inputs)

    def calibrate(self, program: HypothesisProgram, train_observations):
        """Fit the multiplicative constant k so that k * structure(x) ≈ target.
        Uses the geometric mean of target/structure ratios (robust for laws
        spanning orders of magnitude, e.g. the gravitational constant)."""
        import math
        if _skip_numeric_calibration_for_bounded_transform(program, train_observations):
            program._const = 1.0  # type: ignore[attr-defined]
            return program
        ratios = []
        for obs in train_observations:
            try:
                base = float(program.fn(obs.inputs))
                tgt = float(obs.target)
                if base != 0 and base == base and tgt == tgt and tgt != 0:
                    r = tgt / base
                    if r > 0:
                        ratios.append(math.log(r))
            except Exception:
                continue
        if ratios:
            k = math.exp(sum(ratios) / len(ratios))
            program._const = k  # type: ignore[attr-defined]
        else:
            program._const = 1.0  # type: ignore[attr-defined]
        return program

    def loss(self, prediction: Any, obs: Observation) -> float:
        try:
            pred = float(prediction)
            tgt = float(obs.target)
        except Exception:
            return 1.0
        if pred != pred:  # NaN
            return 1.0
        # Relative log loss, squashed to [0,1]
        denom = abs(tgt) + 1e-9
        rel = abs(pred - tgt) / denom
        return min(1.0, rel)

    def render_report(self, winners, observations) -> str:
        if not winners:
            return ""
        best, _ = winners[0]
        return f"{best.description}\n\nLaw: {best.code}"


# ===========================================================================
# 3. DiscoveryBench — statistical analyzer over a table
# ===========================================================================

DEFAULT_DISCOVERY_MODULES = frozenset(
    {
        "universal_kernel",
        "evidence_contract",
        "answer_plan",
        "contrastive_world",
        "problem_frame",
        "answer_slot",
        "slot_contract",
        "temporal",
        "workbench",
        "llm_synthesis",
    }
)


class DiscoveryAdapter(CPIAdapter):
    name = "discoverybench"

    def __init__(self, df, question: str, domain_knowledge: str,
                 column_descriptions: dict[str, str], judge_model: str = "openai/gpt-4o",
                 enabled_modules: set[str] | None = None):
        self.df = df
        self.question = question
        self.domain_knowledge = domain_knowledge
        self.column_descriptions = column_descriptions or {}
        self.judge_model = judge_model
        self.enabled_modules = set(DEFAULT_DISCOVERY_MODULES if enabled_modules is None else enabled_modules)
        self._observations: list[Observation] | None = None

    def interface_description(self) -> str:
        import pandas as pd  # noqa
        cols = []
        for c in self.df.columns:
            desc = self.column_descriptions.get(c, "")
            cols.append(f"  '{c}' ({self.df[c].dtype}){' — ' + desc if desc else ''}")
        return (
            "Environment: a scientific data table. Discover an analyzer that\n"
            "extracts the statistical evidence needed to answer a research question.\n"
            f"RESEARCH QUESTION: {self.question}\n"
            f"DOMAIN CONTEXT: {self.domain_knowledge[:400]}\n"
            "DataFrame `df` columns:\n" + "\n".join(cols) +
            "\n\nThe analyzer receives a pandas DataFrame `df` and returns a dict "
            "{'evidence': str, 'statistic': float}. pandas (pd) and numpy (np) are available."
        )

    def signature_hint(self) -> str:
        return "def analyze(df) -> dict  # returns {'evidence': str, 'statistic': float}"

    def sandbox_globals(self) -> dict[str, Any]:
        import pandas as pd
        import numpy as np
        return {"pd": pd, "np": np, "df": self.df}

    def seed_program_sources(self) -> list[Any]:
        baselines = generate_dataframe_analyzer_baselines(
            self.df,
            question=self.question,
            column_descriptions=self.column_descriptions,
        )
        return [
            {
                "name": baseline.name,
                "description": baseline.rationale,
                "complexity": 1.1,
                "code": baseline.code,
            }
            for baseline in baselines
        ]

    def collect_observations(self) -> list[Observation]:
        # DiscoveryBench has no per-row refutation target during search.
        # We use a SELF-CONSISTENCY refutation: split the table into folds and
        # require the analyzer to produce a stable statistic across folds.
        if self._observations is not None:
            return self._observations
        n = len(self.df)
        if n < 4:
            self._observations = [Observation(inputs=self.df, target=None, context={})]
            return self._observations
        # Two halves: an analyzer whose evidence is real should be stable across both.
        half = n // 2
        obs = [
            Observation(inputs=self.df.iloc[:half], target="fold_a", context={"fold": "a"}),
            Observation(inputs=self.df.iloc[half:], target="fold_b", context={"fold": "b"}),
            Observation(inputs=self.df, target="full", context={"fold": "full"}),
        ]
        self._observations = obs
        return obs

    def execute(self, program: HypothesisProgram, obs: Observation) -> Any:
        return program.fn(obs.inputs)

    def loss(self, prediction: Any, obs: Observation) -> float:
        # Loss = does the analyzer produce question-grounded scientific
        # evidence, not merely any finite statistic. DiscoveryBench judges
        # semantic hypothesis slots, so shape/mean-only analyzers must not get
        # perfect internal loss.
        if not isinstance(prediction, dict):
            return 1.0
        evidence = str(prediction.get("evidence", "") or "")
        if not evidence.strip():
            return 1.0
        try:
            statistic = float(prediction.get("statistic", 0.0))
            if statistic != statistic:
                return 1.0
        except Exception:
            pass
        quality = self._semantic_evidence_quality(prediction)
        return max(0.0, min(1.0, 1.0 - quality))

    def _semantic_evidence_quality(self, prediction: dict) -> float:
        evidence = str(prediction.get("evidence", "") or "")
        low = evidence.lower()
        question_tokens = _semantic_tokens(self.question)
        schema_text = " ".join(
            f"{col} {desc}" for col, desc in self.column_descriptions.items()
        )
        schema_tokens = _semantic_tokens(schema_text)
        evidence_tokens = _semantic_tokens(evidence)
        query_overlap = len(evidence_tokens & question_tokens)
        schema_overlap = len(evidence_tokens & schema_tokens)
        variables = prediction.get("variables", ())
        if isinstance(variables, str):
            variables = [variables]
        variable_tokens = _semantic_tokens(" ".join(str(v) for v in variables))
        role_hits = sum(
            1
            for key in ("cause", "mediator", "outcome", "relation")
            if str(prediction.get(key, "") or "").strip()
        )
        operator_hits = sum(
            1
            for word in (
                "corr", "correlation", "association", "trend", "increase",
                "decrease", "positive", "negative", "mediator", "mediated",
                "effect", "stable", "split", "robust", "maximum", "minimum",
                "growth", "change",
            )
            if word in low
        )
        shape_only = (
            ("rows=" in low or "columns=" in low or "shape" in low)
            and query_overlap == 0
            and schema_overlap == 0
            and role_hits == 0
        )
        if shape_only:
            return 0.0
        score = 0.0
        score += min(0.30, 0.08 * query_overlap)
        score += min(0.20, 0.04 * schema_overlap)
        score += min(0.20, 0.05 * len(variable_tokens & (question_tokens | schema_tokens)))
        score += min(0.20, 0.06 * role_hits)
        score += min(0.20, 0.04 * operator_hits)
        try:
            if abs(float(prediction.get("statistic", 0.0) or 0.0)) > 1e-12:
                score += 0.05
        except Exception:
            pass
        if query_overlap == 0 and schema_overlap == 0 and role_hits == 0:
            score = min(score, 0.35)
        return max(0.0, min(1.0, score))

    def render_report(self, winners, observations) -> str:
        # Collect evidence from all winning analyzers run on the full table,
        # then let the LLM synthesize a hypothesis. The evidence is DISCOVERED;
        # the LLM only verbalizes it (no rubric, no answer key).
        evidence_pieces = []
        for prog, score in winners:
            try:
                out = prog.fn(self.df)
                if isinstance(out, dict) and out.get("evidence"):
                    evidence_pieces.append(f"[{prog.name}] {out['evidence']}")
            except Exception:
                continue
        if "universal_kernel" in self.enabled_modules:
            kernel_result = UniversalHypothesisKernel().run(
                KernelTask(
                    task_text=self.question,
                    interface_kind="discovery_table",
                    data=self.df,
                    domain_context=self.domain_knowledge,
                    schema=self.column_descriptions,
                    metadata={},
                )
            )
            if kernel_result.best is not None and kernel_result.best.coverage >= 0.5:
                return self._final_answer_report(kernel_result.best.report())
        contract = compile_answer_contract(self.question)
        contract_report = self._render_contract_first_report(contract)
        if contract_report:
            return contract_report
        if "evidence_contract" in self.enabled_modules:
            evidence_contract = infer_evidence_contract_hypothesis(
                question=self.question,
                domain_context=self.domain_knowledge,
                data=self.df,
                schema=self.column_descriptions,
                interface_kind="discovery_table",
            )
            if evidence_contract is not None and evidence_contract.coverage.score >= 0.75:
                return self._final_answer_report(evidence_contract.as_report())
        if "answer_plan" in self.enabled_modules:
            answer_plan = infer_answer_plan_hypothesis(
                question=self.question,
                domain_context=self.domain_knowledge,
                df=self.df,
                column_descriptions=self.column_descriptions,
            )
            if answer_plan is not None:
                return self._final_answer_report(
                    f"HYPOTHESIS: {answer_plan.hypothesis}\n"
                    f"WORKFLOW SUMMARY: {answer_plan.workflow} "
                    f"Evidence: {answer_plan.evidence}."
                )
        if "answer_slot" in self.enabled_modules:
            answer_slot = infer_universal_slot_hypothesis(
                task_text=self.question,
                interface_kind="discovery_table",
                data=self.df,
                domain_context=self.domain_knowledge,
                schema=self.column_descriptions,
            )
            if answer_slot is not None:
                return self._final_answer_report(
                    f"HYPOTHESIS: {answer_slot.hypothesis}\n"
                    f"WORKFLOW SUMMARY: {answer_slot.workflow} "
                    f"Evidence: {answer_slot.evidence}."
                )
        if "contrastive_world" in self.enabled_modules:
            contrastive = infer_contrastive_world_hypothesis(
                question=self.question,
                domain_context=self.domain_knowledge,
                data=self.df,
                column_descriptions=self.column_descriptions,
            )
            if contrastive is not None:
                return self._final_answer_report(
                    f"HYPOTHESIS: {contrastive.hypothesis}\n"
                    f"WORKFLOW SUMMARY: {contrastive.workflow} "
                    f"Evidence: {contrastive.evidence}."
                )
        if "problem_frame" in self.enabled_modules:
            problem_frame = infer_problem_frame_hypothesis(
                question=self.question,
                domain_context=self.domain_knowledge,
                data=self.df,
                column_descriptions=self.column_descriptions,
            )
            if problem_frame is not None:
                return self._final_answer_report(
                    f"HYPOTHESIS: {problem_frame.hypothesis}\n"
                    f"WORKFLOW SUMMARY: {problem_frame.workflow} "
                    f"Evidence: {problem_frame.evidence}."
                )
        if "slot_contract" in self.enabled_modules:
            slot_contract = infer_slot_contract_hypothesis(
                question=self.question,
                domain_context=self.domain_knowledge,
                df=self.df,
                column_descriptions=self.column_descriptions,
            )
            if slot_contract is not None:
                return self._final_answer_report(
                    f"HYPOTHESIS: {slot_contract.hypothesis}\n"
                    f"WORKFLOW SUMMARY: {slot_contract.workflow} "
                    f"Evidence: {slot_contract.evidence}."
                )
        if "temporal" in self.enabled_modules:
            temporal = infer_temporal_event_hypothesis(
                question=self.question,
                domain_context=self.domain_knowledge,
                df=self.df,
                column_descriptions=self.column_descriptions,
            )
            if temporal is not None:
                hypothesis, workflow, evidence = temporal
                return self._final_answer_report(
                    f"HYPOTHESIS: {hypothesis}\n"
                    f"WORKFLOW SUMMARY: {workflow} Evidence: {evidence}."
                )
        semantic_report = self._render_semantic_winner_report(winners)
        if semantic_report and contract.answer_type in {"relation", "hypothesis"}:
            return self._final_answer_report(semantic_report)
        if not evidence_pieces:
            return "No stable evidence found."
        if "workbench" in self.enabled_modules:
            workbench = compile_hypothesis_workbench(
                question=self.question,
                domain_context=self.domain_knowledge,
                evidence_lines=evidence_pieces,
            )
            best = workbench.best
            if best is not None:
                return self._final_answer_report(
                    f"HYPOTHESIS: {best.hypothesis}\n"
                    f"WORKFLOW SUMMARY: {best.workflow}"
                )
        if "llm_synthesis" not in self.enabled_modules:
            return self._final_answer_report(
                "\n\n".join(evidence_pieces[:2]).strip() or "No stable evidence found."
            )
        client = make_openai_client()
        prompt = (
            f"Research question: {self.question}\n\n"
            f"Domain context: {self.domain_knowledge[:300]}\n\n"
            f"Evidence discovered by data analyzers:\n" +
            "\n".join(f"- {e}" for e in evidence_pieces[:8]) +
            "\n\nWrite a 1-2 sentence hypothesis answering the question using ONLY "
            "this evidence. Name specific variables/values/periods."
        )
        return self._final_answer_report(call_llm(
            client, model=self.judge_model,
            system="You write evidence-grounded scientific hypotheses.",
            user=prompt, max_tokens=250, temperature=0.2,
        ))

    def _final_answer_report(self, report: str) -> str:
        text = str(report or "").strip()
        match = re.search(
            r"(?is)\bHYPOTHESIS\s*:\s*(.*?)(?:\n\s*WORKFLOW\s+SUMMARY\s*:\s*(.*)|\Z)",
            text,
        )
        if not match:
            return normalize_answer_report(self.question, text)
        hypothesis = normalize_answer_report(self.question, match.group(1).strip())
        workflow = (match.group(2) or "").strip()
        if workflow:
            return f"HYPOTHESIS: {hypothesis}\nWORKFLOW SUMMARY: {workflow}"
        return hypothesis

    def _render_contract_first_report(self, contract) -> str:
        """Close required answer slots before generic evidence can compete."""

        if contract.answer_type not in {"temporal_century", "temporal_event", "entity_or_slot", "relation"}:
            return ""
        slot_contract = infer_slot_contract_hypothesis(
            question=self.question,
            domain_context=self.domain_knowledge,
            df=self.df,
            column_descriptions=self.column_descriptions,
        )
        if slot_contract is None:
            return ""
        slots = dict(slot_contract.slots or {})
        if contract.answer_type in {"temporal_century", "temporal_event"} and "time" not in slots:
            return ""
        if contract.answer_type == "entity_or_slot" and not any(k in slots for k in ("variable", "source", "target", "event")):
            return ""
        if contract.answer_type == "relation":
            event = str(slots.get("event", ""))
            if event not in {
                "window_relationship",
                "pca_component",
                "simultaneous_decline",
                "simultaneous_inverse",
                "peak_context_change",
            }:
                return ""
        return self._final_answer_report(
            f"HYPOTHESIS: {slot_contract.hypothesis}\n"
            f"WORKFLOW SUMMARY: {slot_contract.workflow} "
            f"Evidence: {slot_contract.evidence}."
        )

    def _render_semantic_winner_report(self, winners) -> str:
        """Render structured analyzer output before fallback modules.

        If an executable analyzer has already bound cause/mediator/outcome
        slots, use that measured structure directly.  This prevents generic
        fallback modules from overriding the current best refutable evidence.
        """

        for prog, _score in winners[:5]:
            try:
                out = prog.fn(self.df)
            except Exception:
                continue
            if not isinstance(out, dict):
                continue
            cause = str(out.get("cause", "") or "").strip()
            outcome = str(out.get("outcome", "") or "").strip()
            mediator = str(out.get("mediator", "") or "").strip()
            relation = str(out.get("relation", "") or "").strip()
            evidence = str(out.get("evidence", "") or "").strip()
            if not cause or not outcome or not evidence:
                continue
            if mediator:
                hypothesis = (
                    f"{cause} is associated with {outcome}, with {mediator} acting "
                    f"as an intermediate mechanism or proxy in the observed data."
                )
                workflow = (
                    f"Selected query-relevant variables {cause}, {mediator}, and "
                    f"{outcome}; tested a mediated chain; relation={relation or 'mediated association'}."
                )
            else:
                hypothesis = (
                    f"{cause} is associated with {outcome} in the observed data."
                )
                workflow = (
                    f"Selected query-relevant variables {cause} and {outcome}; "
                    f"tested their association; relation={relation or 'association'}."
                )
            return (
                f"HYPOTHESIS: {hypothesis}\n"
                f"WORKFLOW SUMMARY: {workflow} Evidence: {evidence}."
            )
        return ""


# ===========================================================================
# 4. UltraHorizon Bio — hidden inheritance rules (AUTONOMOUS, no rubric)
# ===========================================================================

class BioAdapter(CPIAdapter):
    name = "uh_bio"

    def __init__(self, cross_records: list[dict], judge_model: str = "openai/gpt-4o"):
        # cross_records: list of conduct_cross results with parents + offspring
        self.cross_records = cross_records
        self.judge_model = judge_model
        self._observations: list[Observation] | None = None

    def interface_description(self) -> str:
        return (
            "Environment: organisms with hidden inheritance rules.\n"
            "Each organism has phenotype traits: body_size (float size_score),\n"
            "body_color (discrete), shell_shape (discrete).\n"
            "A cross of two parents yields offspring with phenotypes, plus a\n"
            "viability_rate (fraction of fertilizations that survived).\n"
            "Propose a predictor: given two parent phenotypes, predict an offspring\n"
            "phenotype property. The predictor is scored against REAL observed\n"
            "offspring from held-out crosses.\n"
            "Parent dicts have keys: body_size, size_score, body_color, shell_shape."
        )

    def signature_hint(self) -> str:
        return (
            "def predict(parent1: dict, parent2: dict, context: dict) -> dict\n"
            "  # return {'body_color': str, 'shell_shape': str, 'size_score': float}\n"
            "  # predicting the MOST COMMON offspring phenotype"
        )

    def collect_observations(self) -> list[Observation]:
        if self._observations is not None:
            return self._observations
        obs = []
        for cr in self.cross_records:
            p1 = cr.get("parent1_phenotype", {})
            p2 = cr.get("parent2_phenotype", {})
            offspring = cr.get("offspring", [])
            if not offspring:
                continue
            # Target = most common offspring phenotype + mean size + viability
            from collections import Counter
            colors = Counter(o.get("phenotype", {}).get("body_color") for o in offspring)
            shells = Counter(o.get("phenotype", {}).get("shell_shape") for o in offspring)
            sizes = [o.get("phenotype", {}).get("size_score", 0) for o in offspring]
            target = {
                "body_color": colors.most_common(1)[0][0] if colors else None,
                "shell_shape": shells.most_common(1)[0][0] if shells else None,
                "size_score": sum(sizes) / len(sizes) if sizes else 0.0,
                "viability_rate": cr.get("viability_rate", 1.0),
            }
            obs.append(Observation(
                inputs={"parent1": p1, "parent2": p2},
                target=target,
                context={"viability_rate": cr.get("viability_rate", 1.0)},
            ))
        self._observations = obs
        return obs

    def execute(self, program: HypothesisProgram, obs: Observation) -> Any:
        return program.fn(obs.inputs["parent1"], obs.inputs["parent2"], obs.context)

    def loss(self, prediction: Any, obs: Observation) -> float:
        if not isinstance(prediction, dict):
            return 1.0
        tgt = obs.target
        parts = []
        # Color match
        if tgt.get("body_color") is not None:
            parts.append(0.0 if prediction.get("body_color") == tgt["body_color"] else 1.0)
        # Shell match
        if tgt.get("shell_shape") is not None:
            parts.append(0.0 if prediction.get("shell_shape") == tgt["shell_shape"] else 1.0)
        # Size relative error
        if tgt.get("size_score"):
            try:
                pred_s = float(prediction.get("size_score", 0))
                rel = abs(pred_s - tgt["size_score"]) / (abs(tgt["size_score"]) + 1e-9)
                parts.append(min(1.0, rel))
            except Exception:
                parts.append(1.0)
        return sum(parts) / len(parts) if parts else 1.0

    def render_report(self, winners, observations) -> str:
        # Verbalize the DISCOVERED predictor programs into an inheritance report.
        # The LLM sees the winning programs' code + the raw cross data, and writes
        # a report. It has NO access to the scoring rubric.
        if not winners:
            return "No predictive inheritance rule found."
        program_summaries = []
        for prog, score in winners[:3]:
            program_summaries.append(
                f"Program '{prog.name}' (prediction loss {score.loss_mean:.2f}):\n"
                f"{prog.description}\n```python\n{prog.code}\n```"
            )
        # Provide raw cross statistics too (discovered facts)
        facts = _bio_facts(self.cross_records)
        client = make_openai_client()
        prompt = (
            "You are a geneticist writing an inheritance report for an alien organism.\n"
            "You have discovered predictive programs by experiment, and raw cross data.\n\n"
            f"DISCOVERED PREDICTORS:\n" + "\n\n".join(program_summaries) + "\n\n"
            f"RAW EXPERIMENTAL FACTS:\n{facts}\n\n"
            "Write a formal inheritance report. Cover: genetic architecture (ploidy if\n"
            "inferable from viability patterns), each trait's inheritance mechanism\n"
            "(size, color, shell), any dominance ordering, dosage effects, and lethal\n"
            "combinations evident from low viability. Base everything on the data above."
        )
        return call_llm(
            client, model=self.judge_model,
            system="You write rigorous genetics reports grounded only in given data.",
            user=prompt, max_tokens=900, temperature=0.3,
        )


# ===========================================================================
# Helpers
# ===========================================================================

def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (0 if ca == cb else 1)))
        prev = cur
    return prev[-1]


def _skip_numeric_calibration_for_bounded_transform(
    program: HypothesisProgram,
    train_observations,
) -> bool:
    text = f"{program.name}\n{program.description}\n{program.code}".lower()
    if not (("asin" in text or "acos" in text) and "sin" in text):
        return False
    targets = []
    preds = []
    for obs in train_observations:
        try:
            targets.append(float(obs.target))
            preds.append(float(program.fn(obs.inputs)))
        except Exception:
            continue
    if len(targets) < 3 or not preds:
        return False
    target_constant = max(targets) - min(targets) <= 1e-9
    pred_nonconstant = max(preds) - min(preds) > 1e-9
    return target_constant and pred_nonconstant


def _semantic_tokens(text: str) -> set[str]:
    import re

    stop = {
        "the", "and", "for", "with", "from", "into", "between", "among", "which",
        "what", "when", "where", "does", "did", "were", "was", "are", "how",
        "there", "this", "that", "have", "has", "had", "over", "under", "after",
        "before", "during", "in", "of", "to", "a", "an", "on", "by", "as",
        "using", "use", "used", "table", "data", "dataset", "variable",
        "variables", "column", "columns", "row", "rows",
    }
    return {
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_/-]{2,}", str(text).lower())
        if token not in stop
    }


def _bio_facts(cross_records: list[dict]) -> str:
    """Summarize raw, DISCOVERED experimental facts (no rubric knowledge)."""
    from collections import Counter
    lines = []
    viabilities = [cr.get("viability_rate", 1.0) for cr in cross_records]
    if viabilities:
        lines.append(f"- viability rates across {len(viabilities)} crosses: "
                     f"min={min(viabilities):.2f}, max={max(viabilities):.2f}, "
                     f"mean={sum(viabilities)/len(viabilities):.2f}")
    # Size range
    all_sizes = []
    for cr in cross_records:
        for o in cr.get("offspring", []):
            s = o.get("phenotype", {}).get("size_score")
            if s:
                all_sizes.append(s)
    # Parent sizes
    for cr in cross_records[:1]:
        for pk in ("parent1_phenotype", "parent2_phenotype"):
            p = cr.get(pk, {})
            if p:
                lines.append(f"- example parent: size={p.get('size_score')}, "
                             f"color={p.get('body_color')}, shell={p.get('shell_shape')}")
    if all_sizes:
        lines.append(f"- offspring size_score range: {min(all_sizes):.1f} to {max(all_sizes):.1f}")
    # Color/shell transitions: parent colors -> dominant offspring color
    for cr in cross_records[:8]:
        p1 = cr.get("parent1_phenotype", {})
        p2 = cr.get("parent2_phenotype", {})
        offspring = cr.get("offspring", [])
        if not offspring:
            continue
        colors = Counter(o.get("phenotype", {}).get("body_color") for o in offspring)
        shells = Counter(o.get("phenotype", {}).get("shell_shape") for o in offspring)
        dom_c = colors.most_common(1)[0][0] if colors else "?"
        dom_s = shells.most_common(1)[0][0] if shells else "?"
        lines.append(
            f"- cross {p1.get('body_color')}/{p1.get('shell_shape')} x "
            f"{p2.get('body_color')}/{p2.get('shell_shape')} → "
            f"offspring mostly {dom_c}/{dom_s}, viability {cr.get('viability_rate',1.0):.2f}"
        )
    return "\n".join(lines)
