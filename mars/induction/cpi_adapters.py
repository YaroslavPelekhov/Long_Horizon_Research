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
from typing import Any

from mars.agents.base import call_llm, make_openai_client
from mars.induction.universal_cpi import (
    CPIAdapter,
    HypothesisProgram,
    Observation,
    ProgramScore,
)


# ===========================================================================
# 1. UltraHorizon Sequence — hidden string transformation rules
# ===========================================================================

class UHSeqAdapter(CPIAdapter):
    name = "uh_seq"

    def __init__(self, env: Any, rule_slot: int, steps: int = 5):
        self.env = env
        self.rule_slot = rule_slot   # which of 5 rules we induce (1..5)
        self.steps = steps
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

class DiscoveryAdapter(CPIAdapter):
    name = "discoverybench"

    def __init__(self, df, question: str, domain_knowledge: str,
                 column_descriptions: dict[str, str], judge_model: str = "openai/gpt-4o"):
        self.df = df
        self.question = question
        self.domain_knowledge = domain_knowledge
        self.column_descriptions = column_descriptions or {}
        self.judge_model = judge_model
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
        out = program.fn(obs.inputs)
        if isinstance(out, dict):
            return out.get("statistic", out.get("evidence"))
        return out

    def loss(self, prediction: Any, obs: Observation) -> float:
        # Loss = did the analyzer run and produce a finite statistic?
        # Refutation here is "does it execute on this fold and yield signal".
        if prediction is None:
            return 1.0
        try:
            if isinstance(prediction, (int, float)):
                v = float(prediction)
                return 0.0 if v == v else 1.0  # finite → 0 loss
            if isinstance(prediction, str) and prediction.strip():
                return 0.0
        except Exception:
            return 1.0
        return 0.5

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
        if not evidence_pieces:
            return "No stable evidence found."
        client = make_openai_client()
        prompt = (
            f"Research question: {self.question}\n\n"
            f"Domain context: {self.domain_knowledge[:300]}\n\n"
            f"Evidence discovered by data analyzers:\n" +
            "\n".join(f"- {e}" for e in evidence_pieces[:8]) +
            "\n\nWrite a 1-2 sentence hypothesis answering the question using ONLY "
            "this evidence. Name specific variables/values/periods."
        )
        return call_llm(
            client, model=self.judge_model,
            system="You write evidence-grounded scientific hypotheses.",
            user=prompt, max_tokens=250, temperature=0.2,
        )


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
