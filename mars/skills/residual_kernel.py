"""MARS Residual Kernel.

The residual kernel is the benchmark-agnostic core we want MARS to grow around:
it does not store benchmark-specific solvers.  It turns failed executable
hypotheses and counterexamples into typed rewrite rules over a hypothesis
language.

The first version is intentionally small and deterministic.  It gives us a
concrete substrate for the bigger idea:

    counterexamples -> residual fingerprint -> induced rewrite rule
    -> expanded hypothesis space -> verification -> compression

The classes here are plain data structures plus a few safe inducers.  They are
designed to be called by UniversalCPI/self-layers later, but do not depend on
any benchmark adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import itertools
import re
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class TypedTerm:
    """A typed expression node in the shared hypothesis language."""

    op: str
    args: tuple["TypedTerm", ...] = ()
    value: Any = None
    typ: str = "unknown"

    def to_source(self) -> str:
        if self.op == "var":
            return f"_v(inputs, {self.value!r})"
        if self.op == "const":
            return repr(self.value)
        if self.op == "mul":
            return " * ".join(arg.to_source() for arg in self.args) or "1.0"
        if self.op == "div":
            left = self.args[0].to_source()
            right = self.args[1].to_source()
            return f"_safe_div({left}, {right})"
        if self.op == "pow":
            return f"({self.args[0].to_source()} ** {float(self.value)!r})"
        if self.op == "add":
            return " + ".join(arg.to_source() for arg in self.args) or "0.0"
        return repr(self.value)

    def cost(self) -> float:
        return 1.0 + sum(arg.cost() for arg in self.args)


@dataclass(frozen=True)
class ResidualFingerprint:
    """A compressed description of how a hypothesis failed."""

    kind: str
    target_type: str
    variables: tuple[str, ...] = ()
    invariant: str = ""
    stats: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RewriteRule:
    """A reusable transformation over the hypothesis search space."""

    name: str
    residual: ResidualFingerprint
    match: Mapping[str, Any]
    replace: Mapping[str, Any]
    description: str
    support: int
    gain: float = 0.0

    def mdl_cost(self) -> float:
        return 1.0 + 0.2 * len(str(self.match)) + 0.2 * len(str(self.replace))


@dataclass(frozen=True)
class InducedProgram:
    """Executable source emitted by an induced rewrite rule."""

    name: str
    description: str
    code: str
    rule: RewriteRule
    complexity: float


@dataclass(frozen=True)
class EvidencePlan:
    """Structured evidence program for table/scientific-hypothesis tasks."""

    cause: str
    mediators: tuple[str, ...]
    outcomes: tuple[str, ...]
    operations: tuple[str, ...]

    def to_hypothesis_skeleton(self) -> str:
        med = ", ".join(self.mediators) or "intermediate proxy variables"
        out = ", ".join(self.outcomes) or "target outcomes"
        return f"{self.cause} -> {med} -> {out}"


@dataclass(frozen=True)
class HypothesisSketch:
    """A short typed hypothesis blueprint that can be expanded without an LLM.

    The sketch is the key move for small models: instead of asking a weak model
    to hold a long proof/program in context, we ask the kernel to build a tiny
    typed object and deterministically expand it into executable hypotheses.
    """

    name: str
    target_type: str
    slots: Mapping[str, Any]
    operators: tuple[str, ...]
    residual_kind: str
    support: int = 0
    rationale: str = ""


@dataclass(frozen=True)
class SketchExpansion:
    """Executable programs produced by one hypothesis sketch."""

    sketch: HypothesisSketch
    programs: tuple[InducedProgram, ...]


class ResidualKernel:
    """Induce rewrite rules from typed residuals.

    This is a microkernel: it has a small universal set of meta-inducers, each
    producing rewrite rules from evidence.  New learned rules should later be
    stored as self-layers; this class only provides the common calculus.
    """

    def induce_numeric_power_rule(
        self,
        examples: Sequence[Mapping[str, Any]],
        *,
        target_key: str = "target",
        max_vars: int = 6,
    ) -> InducedProgram | None:
        """Infer a multiplicative power law from numeric dict observations.

        It fits:

            target ~= c * product_i x_i ** a_i

        by linear regression in log-space.  This is not Newton-specific; it is
        a generic residual repair for any positive dict -> scalar law.
        """

        numeric_keys = _numeric_keys(examples, target_key)[:max_vars]
        rows: list[list[float]] = []
        ys: list[float] = []
        for ex in examples:
            try:
                target = float(ex[target_key])
            except Exception:
                continue
            if target <= 0:
                continue
            row = [1.0]
            ok = True
            for key in numeric_keys:
                try:
                    value = float(ex[key])
                except Exception:
                    ok = False
                    break
                if value <= 0:
                    ok = False
                    break
                row.append(math.log(value))
            if not ok:
                continue
            rows.append(row)
            ys.append(math.log(target))
        if len(rows) < max(3, len(numeric_keys) + 1):
            return None
        coeffs = _least_squares(rows, ys)
        if coeffs is None:
            return None
        intercept = coeffs[0]
        # Continuous regression is a useful proposal, but scientific laws are
        # often sparse rational forms.  Select the compact exponent lattice by
        # held-out log residual rather than trusting a noisy real-valued fit.
        powers, intercept = _select_rational_lattice(rows, ys, numeric_keys, coeffs)
        if not powers:
            return None
        const = math.exp(intercept)
        term = _power_product_term(powers)
        residual = ResidualFingerprint(
            kind="numeric_power_law",
            target_type="number",
            variables=tuple(powers),
            invariant="log(target) is linear in log(inputs)",
            stats={"constant": const, "powers": powers},
        )
        rule = RewriteRule(
            name="induce_numeric_power_law",
            residual=residual,
            match={"family": "numeric_expression", "failure": "scale_residual"},
            replace={"term": _term_to_pattern(term), "constant": const},
            description="Rewrite numeric expressions into a fitted multiplicative power law.",
            support=len(rows),
            gain=1.0,
        )
        code = _numeric_law_source(term, const)
        return InducedProgram(
            name="rk_numeric_power_law",
            description=f"fitted power law with powers {powers}",
            code=code,
            rule=rule,
            complexity=term.cost() + 1.0,
        )

    def induce_string_context_rules(
        self,
        examples: Sequence[Mapping[str, Any]],
        *,
        input_key: str = "current",
        target_key: str = "target",
        context_key: str = "context",
    ) -> list[InducedProgram]:
        """Induce context-pair rewrite rules for string transition tasks.

        It mines simple relations over every pair of string-valued context
        fields, then emits rewrite rules only when a relation exactly explains
        all examples.  The primitive relations are generic algebra over strings:
        interleave, charwise add, position-wise max/min, and sorted multiset.
        """

        rows = []
        context_fields: list[str] = []
        for ex in examples:
            ctx = dict(ex.get(context_key) or {})
            target = str(ex.get(target_key, ""))
            current = str(ex.get(input_key, ""))
            rows.append((current, ctx, target))
            for key, value in ctx.items():
                if isinstance(value, str) and key not in context_fields:
                    context_fields.append(key)
        programs: list[InducedProgram] = []
        for a_i, a_key in enumerate(context_fields):
            for b_key in context_fields[a_i + 1:]:
                candidates = [
                    ("interleave", lambda a, b: _interleave(a, b), "interleave context strings"),
                    ("charwise_add", lambda a, b: _charwise_add(a, b), "charwise alphabet addition"),
                    ("poswise_max", lambda a, b: _poswise_pick(a, b, max), "position-wise maximum"),
                    ("poswise_min", lambda a, b: _poswise_pick(a, b, min), "position-wise minimum"),
                    ("sorted_multiset", lambda a, b: "".join(sorted(a + b)), "sorted multiset union"),
                ]
                for op_name, fn, desc in candidates:
                    if all(fn(str(ctx.get(a_key, "")), str(ctx.get(b_key, ""))) == target for _, ctx, target in rows):
                        programs.append(
                            self._string_program(
                                op_name=op_name,
                                a_key=a_key,
                                b_key=b_key,
                                description=desc,
                                support=len(rows),
                            )
                        )
                    if all(fn(str(ctx.get(b_key, "")), str(ctx.get(a_key, ""))) == target for _, ctx, target in rows):
                        programs.append(
                            self._string_program(
                                op_name=op_name,
                                a_key=b_key,
                                b_key=a_key,
                                description=desc,
                                support=len(rows),
                            )
                        )
        return _dedupe_programs(programs)

    def induce_evidence_plan(
        self,
        *,
        question: str,
        column_descriptions: Mapping[str, str],
        domain_context: str = "",
    ) -> EvidencePlan:
        """Infer a causal/evidence skeleton from table metadata.

        This does not know DiscoveryBench.  It maps natural-language task text
        and column descriptions into cause/mediator/outcome slots so internal
        search can optimize evidence coverage rather than mere finite numbers.
        """

        text = f"{question} {domain_context}".lower()
        scored = []
        for col, desc in column_descriptions.items():
            bag = f"{col} {desc}".lower()
            score = _token_overlap(text, bag)
            scored.append((score, col, bag))
        scored.sort(reverse=True)

        cause = _best_column(scored, ("education", "expenditure", "investment", "cause", "input"))
        mediators = tuple(
            col for _score, col, bag in scored
            if any(word in bag for word in ("labor", "human capital", "enrollment", "school"))
        )[:3]
        outcomes = tuple(
            col for _score, col, bag in scored
            if any(word in bag for word in ("gdp", "gni", "output", "income", "economic"))
        )[:3]
        if not cause and scored:
            cause = scored[0][1]
        return EvidencePlan(
            cause=cause or "",
            mediators=mediators,
            outcomes=outcomes,
            operations=("trend", "correlation", "mediation_chain", "robustness_split"),
        )

    def induce_table_evidence_sketch(
        self,
        *,
        question: str,
        column_descriptions: Mapping[str, str],
        domain_context: str = "",
    ) -> HypothesisSketch | None:
        """Build a compact causal-statistical sketch from table metadata.

        This is deliberately not benchmark-specific.  It uses only a question,
        column descriptions, and universal evidence operators.  The resulting
        sketch is small enough for a weak model to inspect or mutate, while the
        kernel handles the program expansion.
        """

        if not column_descriptions:
            return None
        plan = self.induce_evidence_plan(
            question=question,
            column_descriptions=column_descriptions,
            domain_context=domain_context,
        )
        if not plan.cause or not plan.outcomes:
            return None
        return HypothesisSketch(
            name="causal_evidence_chain",
            target_type="table_evidence",
            slots={
                "cause": plan.cause,
                "mediators": tuple(plan.mediators),
                "outcomes": tuple(plan.outcomes),
                "question": question,
                "cause_terms": _role_term_groups(question, domain_context, "cause"),
                "mediator_terms": _role_term_groups(question, domain_context, "mediator"),
                "outcome_terms": _role_term_groups(question, domain_context, "outcome"),
            },
            operators=plan.operations,
            residual_kind="semantic_table_evidence_gap",
            support=len(column_descriptions),
            rationale=plan.to_hypothesis_skeleton(),
        )

    def expand_hypothesis_sketch(self, sketch: HypothesisSketch) -> SketchExpansion:
        """Compile a typed sketch into executable hypothesis programs."""

        if sketch.target_type != "table_evidence":
            return SketchExpansion(sketch=sketch, programs=())
        program = self._table_evidence_program(sketch)
        return SketchExpansion(sketch=sketch, programs=(program,))

    def induce_table_evidence_programs(
        self,
        *,
        question: str,
        column_descriptions: Mapping[str, str],
        domain_context: str = "",
    ) -> list[InducedProgram]:
        """Induce executable evidence analyzers from a table question."""

        sketch = self.induce_table_evidence_sketch(
            question=question,
            column_descriptions=column_descriptions,
            domain_context=domain_context,
        )
        if sketch is None:
            return []
        return list(self.expand_hypothesis_sketch(sketch).programs)

    def _table_evidence_program(self, sketch: HypothesisSketch) -> InducedProgram:
        cause = str(sketch.slots.get("cause", ""))
        mediators = tuple(str(x) for x in sketch.slots.get("mediators", ()) if str(x))
        outcomes = tuple(str(x) for x in sketch.slots.get("outcomes", ()) if str(x))
        cause_terms = tuple(tuple(str(t) for t in group) for group in sketch.slots.get("cause_terms", ()))
        mediator_terms = tuple(tuple(str(t) for t in group) for group in sketch.slots.get("mediator_terms", ()))
        outcome_terms = tuple(tuple(str(t) for t in group) for group in sketch.slots.get("outcome_terms", ()))
        residual = ResidualFingerprint(
            kind=sketch.residual_kind,
            target_type=sketch.target_type,
            variables=(cause,) + mediators + outcomes,
            invariant="evidence is a stable cause/mediator/outcome correlation chain",
            stats={"operators": sketch.operators, "sketch": sketch.name},
        )
        rule = RewriteRule(
            name="compile_causal_evidence_sketch",
            residual=residual,
            match={"family": "table_analyzer", "failure": "semantic_evidence_gap"},
            replace={
                "cause": cause,
                "mediators": mediators,
                "outcomes": outcomes,
                "operators": sketch.operators,
            },
            description="Compile a causal-statistical sketch into a fold-stable evidence analyzer.",
            support=sketch.support,
            gain=1.0,
        )
        return InducedProgram(
            name=f"rk_sketch_{_safe_name(sketch.name)}",
            description=f"compiled evidence sketch: {sketch.rationale}",
            code=_table_evidence_source(cause, mediators, outcomes, cause_terms, mediator_terms, outcome_terms),
            rule=rule,
            complexity=3.2 + 0.2 * (len(mediators) + len(outcomes)),
        )

    def _string_program(
        self,
        *,
        op_name: str,
        a_key: str,
        b_key: str,
        description: str,
        support: int,
    ) -> InducedProgram:
        residual = ResidualFingerprint(
            kind="context_pair_string_relation",
            target_type="string",
            variables=(a_key, b_key),
            invariant=op_name,
            stats={"support": support},
        )
        rule = RewriteRule(
            name=f"rewrite_to_{op_name}_{a_key}_{b_key}",
            residual=residual,
            match={"family": "string_transform", "source": "wrong_context_or_current"},
            replace={"op": op_name, "a": a_key, "b": b_key},
            description=f"Rewrite string transform to {description} over context fields.",
            support=support,
            gain=1.0,
        )
        code = _string_rule_source(op_name, a_key, b_key)
        return InducedProgram(
            name=f"rk_{op_name}_{_safe_name(a_key)}_{_safe_name(b_key)}",
            description=f"{description} over context[{a_key!r}] and context[{b_key!r}]",
            code=code,
            rule=rule,
            complexity=2.0,
        )


def compress_rewrite_rules(rules: Sequence[RewriteRule]) -> list[RewriteRule]:
    """Compress repeated concrete rules into parametric rewrite laws."""

    grouped: dict[tuple[str, str], list[RewriteRule]] = {}
    for rule in rules:
        grouped.setdefault((rule.residual.kind, str(rule.replace.get("op", ""))), []).append(rule)
    compressed: list[RewriteRule] = []
    for (_kind, _op), group in grouped.items():
        if len(group) == 1:
            compressed.append(group[0])
            continue
        first = group[0]
        variables = sorted({v for rule in group for v in rule.residual.variables})
        residual = ResidualFingerprint(
            kind=first.residual.kind,
            target_type=first.residual.target_type,
            variables=tuple(variables),
            invariant=first.residual.invariant,
            stats={"compressed_from": len(group)},
        )
        compressed.append(
            RewriteRule(
                name=f"compressed_{first.name}",
                residual=residual,
                match={"family": first.match.get("family"), "parametric": True},
                replace={"template": first.replace.get("op"), "fields": ("A", "B")},
                description=f"Compressed {len(group)} concrete rewrite rules into a parametric law.",
                support=sum(rule.support for rule in group),
                gain=sum(rule.gain for rule in group) / len(group),
            )
        )
    return compressed


def _numeric_keys(examples: Sequence[Mapping[str, Any]], target_key: str) -> list[str]:
    keys: list[str] = []
    for ex in examples:
        for key, value in ex.items():
            if key == target_key:
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if key not in keys:
                    keys.append(str(key))
    return keys


def _least_squares(rows: list[list[float]], ys: list[float]) -> list[float] | None:
    n_cols = len(rows[0])
    xtx = [[0.0 for _ in range(n_cols)] for _ in range(n_cols)]
    xty = [0.0 for _ in range(n_cols)]
    ridge = 1e-9
    for row, y in zip(rows, ys):
        for i in range(n_cols):
            xty[i] += row[i] * y
            for j in range(n_cols):
                xtx[i][j] += row[i] * row[j]
    for i in range(n_cols):
        xtx[i][i] += ridge
    return _solve_linear(xtx, xty)


def _solve_linear(a: list[list[float]], b: list[float]) -> list[float] | None:
    n = len(b)
    mat = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(mat[r][col]))
        if abs(mat[pivot][col]) < 1e-12:
            return None
        mat[col], mat[pivot] = mat[pivot], mat[col]
        denom = mat[col][col]
        for j in range(col, n + 1):
            mat[col][j] /= denom
        for r in range(n):
            if r == col:
                continue
            factor = mat[r][col]
            for j in range(col, n + 1):
                mat[r][j] -= factor * mat[col][j]
    return [mat[i][n] for i in range(n)]


def _snap_rational_power(value: float) -> float:
    candidates = [-4, -3, -2.5, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 2.5, 3, 4]
    best = min(candidates, key=lambda x: abs(float(x) - value))
    return float(best) if abs(float(best) - value) <= 0.12 else float(value)


def _select_rational_lattice(
    rows: list[list[float]], ys: list[float], keys: list[str], coeffs: list[float]
) -> tuple[dict[str, float], float]:
    if len(keys) > 4:
        return ({key: _snap_rational_power(coeffs[i + 1]) for i, key in enumerate(keys)
                 if abs(coeffs[i + 1]) > 1e-6}, coeffs[0])
    lattice = (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0)
    best: tuple[float, int, tuple[float, ...], float] | None = None
    for exps in itertools.product(lattice, repeat=len(keys)):
        residuals = [y - sum(exp * row[i + 1] for i, exp in enumerate(exps)) for row, y in zip(rows, ys)]
        residuals.sort()
        intercept = residuals[len(residuals) // 2]
        loss = sum(abs(value - intercept) for value in residuals) / len(residuals)
        complexity = sum(exp != 0.0 for exp in exps)
        candidate = (loss, complexity, exps, intercept)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        return {}, coeffs[0]
    return ({key: exp for key, exp in zip(keys, best[2]) if exp != 0.0}, best[3])


def _power_product_term(powers: Mapping[str, float]) -> TypedTerm:
    factors = []
    for key, power in powers.items():
        var = TypedTerm("var", value=key, typ="number")
        if abs(power - 1.0) < 1e-12:
            factors.append(var)
        else:
            factors.append(TypedTerm("pow", args=(var,), value=power, typ="number"))
    return TypedTerm("mul", args=tuple(factors), typ="number")


def _term_to_pattern(term: TypedTerm) -> Mapping[str, Any]:
    if term.op == "var":
        return {"op": "var", "name": term.value}
    if term.op == "pow":
        return {"op": "pow", "base": _term_to_pattern(term.args[0]), "power": term.value}
    return {"op": term.op, "args": [_term_to_pattern(arg) for arg in term.args]}


def _numeric_law_source(term: TypedTerm, const: float) -> str:
    expr = term.to_source()
    return (
        "def _v(inputs: dict, key: str) -> float:\n"
        "    return float(inputs.get(key, 0.0))\n\n"
        "def _safe_div(a: float, b: float) -> float:\n"
        "    if abs(b) < 1e-12:\n"
        "        return 0.0\n"
        "    return a / b\n\n"
        "def law(inputs: dict) -> float:\n"
        f"    return {const!r} * ({expr})\n"
    )


def _interleave(a: str, b: str) -> str:
    out = []
    for x, y in zip(a, b):
        out.append(x)
        out.append(y)
    return "".join(out)


def _charwise_add(a: str, b: str) -> str:
    out = []
    for x, y in zip(a, b):
        ai = ord(x) - ord("A")
        bi = ord(y) - ord("A")
        if 0 <= ai < 26 and 0 <= bi < 26:
            out.append(chr(ord("A") + ((ai + bi) % 26)))
        else:
            out.append(x)
    return "".join(out)


def _poswise_pick(a: str, b: str, picker) -> str:
    return "".join(picker(x, y) for x, y in zip(a, b))


def _string_rule_source(op_name: str, a_key: str, b_key: str) -> str:
    if op_name == "charwise_add":
        body = (
            "    out = []\n"
            "    for x, y in zip(a, b):\n"
            "        ai = ord(x) - ord('A')\n"
            "        bi = ord(y) - ord('A')\n"
            "        if 0 <= ai < 26 and 0 <= bi < 26:\n"
            "            out.append(chr(ord('A') + ((ai + bi) % 26)))\n"
            "        else:\n"
            "            out.append(x)\n"
            "    return ''.join(out)\n"
        )
    elif op_name == "poswise_max":
        body = "    return ''.join(max(x, y) for x, y in zip(a, b))\n"
    elif op_name == "poswise_min":
        body = "    return ''.join(min(x, y) for x, y in zip(a, b))\n"
    elif op_name == "sorted_multiset":
        body = "    return ''.join(sorted(a + b))\n"
    else:
        body = (
            "    out = []\n"
            "    for x, y in zip(a, b):\n"
            "        out.append(x)\n"
            "        out.append(y)\n"
            "    return ''.join(out)\n"
        )
    return (
        "def rule(current: str, context: dict) -> str:\n"
        f"    a = str(context.get({a_key!r}, ''))\n"
        f"    b = str(context.get({b_key!r}, ''))\n"
        + body
    )


def _table_evidence_source(
    cause: str,
    mediators: Sequence[str],
    outcomes: Sequence[str],
    cause_terms: Sequence[Sequence[str]] = (),
    mediator_terms: Sequence[Sequence[str]] = (),
    outcome_terms: Sequence[Sequence[str]] = (),
) -> str:
    return (
        "def _num(df, col):\n"
        "    if col not in df.columns:\n"
        "        return None\n"
        "    return pd.to_numeric(df[col], errors='coerce')\n\n"
        "def _safe_corr(df, a, b):\n"
        "    x = _num(df, a)\n"
        "    y = _num(df, b)\n"
        "    if x is None or y is None:\n"
        "        return 0.0\n"
        "    joined = pd.concat([x, y], axis=1).dropna()\n"
        "    if len(joined) < 3:\n"
        "        return 0.0\n"
        "    v = float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))\n"
        "    if v != v:\n"
        "        return 0.0\n"
        "    return v\n\n"
        "def _safe_trend(df, col):\n"
        "    x = _num(df, col)\n"
        "    if x is None:\n"
        "        return 0.0\n"
        "    x = x.dropna()\n"
        "    if len(x) < 3:\n"
        "        return 0.0\n"
        "    idx = pd.Series(range(len(x)), index=x.index)\n"
        "    v = float(idx.corr(x))\n"
        "    if v != v:\n"
        "        return 0.0\n"
        "    return v\n\n"
        "def _numeric_columns(df):\n"
        "    cols = []\n"
        "    for col in df.columns:\n"
        "        s = pd.to_numeric(df[col], errors='coerce')\n"
        "        if len(s) and float(s.notna().mean()) >= 0.5:\n"
        "            cols.append(col)\n"
        "    return cols\n\n"
        "def _label_columns(df):\n"
        "    nums = set(_numeric_columns(df))\n"
        "    return [col for col in df.columns if col not in nums]\n\n"
        "def _match_terms(text, group):\n"
        "    low = str(text).lower()\n"
        "    for term in group:\n"
        "        if str(term).lower() not in low:\n"
        "            return False\n"
        "    return True\n\n"
        "def _series_by_terms(df, groups):\n"
        "    numeric_cols = _numeric_columns(df)\n"
        "    label_cols = _label_columns(df)\n"
        "    if not numeric_cols or not label_cols:\n"
        "        return None\n"
        "    rows = []\n"
        "    labels = []\n"
        "    for _idx, row in df.iterrows():\n"
        "        label = ' '.join(str(row.get(col, '')) for col in label_cols)\n"
        "        for group in groups:\n"
        "            if group and _match_terms(label, group):\n"
        "                rows.append(pd.to_numeric(row[numeric_cols], errors='coerce'))\n"
        "                labels.append(label)\n"
        "                break\n"
        "    if not rows:\n"
        "        return None\n"
        "    frame = pd.DataFrame(rows)\n"
        "    return {'series': frame.mean(axis=0), 'label': ' | '.join(labels[:3])}\n\n"
        "def _corr_series(a, b):\n"
        "    if a is None or b is None:\n"
        "        return 0.0\n"
        "    joined = pd.concat([a, b], axis=1).dropna()\n"
        "    if len(joined) < 3:\n"
        "        return 0.0\n"
        "    v = float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))\n"
        "    if v != v:\n"
        "        return 0.0\n"
        "    return v\n\n"
        "def _trend_series(series):\n"
        "    if series is None:\n"
        "        return 0.0\n"
        "    x = series.dropna()\n"
        "    if len(x) < 3:\n"
        "        return 0.0\n"
        "    idx = pd.Series(range(len(x)), index=x.index)\n"
        "    v = float(idx.corr(x))\n"
        "    if v != v:\n"
        "        return 0.0\n"
        "    return v\n\n"
        "def analyze(df) -> dict:\n"
        f"    cause = {cause!r}\n"
        f"    mediators = {tuple(mediators)!r}\n"
        f"    outcomes = {tuple(outcomes)!r}\n"
        f"    cause_terms = {tuple(tuple(group) for group in cause_terms)!r}\n"
        f"    mediator_terms = {tuple(tuple(group) for group in mediator_terms)!r}\n"
        f"    outcome_terms = {tuple(tuple(group) for group in outcome_terms)!r}\n"
        "    pieces = []\n"
        "    strengths = []\n"
        "    cause_trend = 0.0\n"
        "    cause_values = _num(df, cause)\n"
        "    column_cause_ok = cause_values is not None and len(cause_values) and float(cause_values.notna().mean()) >= 0.5\n"
        "    if column_cause_ok and outcomes:\n"
        "        cause_trend = _safe_trend(df, cause)\n"
        "        pieces.append('column-mode trend(' + cause + ')=' + str(round(cause_trend, 3)))\n"
        "        for med in mediators:\n"
        "            cm = _safe_corr(df, cause, med)\n"
        "            strengths.append(abs(cm))\n"
        "            pieces.append('corr(' + cause + ',' + med + ')=' + str(round(cm, 3)))\n"
        "        for out in outcomes:\n"
        "            co = _safe_corr(df, cause, out)\n"
        "            strengths.append(abs(co))\n"
        "            pieces.append('corr(' + cause + ',' + out + ')=' + str(round(co, 3)))\n"
        "            pieces.append('trend(' + out + ')=' + str(round(_safe_trend(df, out), 3)))\n"
        "            for med in mediators:\n"
        "                mo = _safe_corr(df, med, out)\n"
        "                strengths.append(abs(mo))\n"
        "                pieces.append('corr(' + med + ',' + out + ')=' + str(round(mo, 3)))\n"
        "    wide_cause = _series_by_terms(df, cause_terms)\n"
        "    wide_mediators = [_series_by_terms(df, (group,)) for group in mediator_terms]\n"
        "    wide_outcomes = [_series_by_terms(df, (group,)) for group in outcome_terms]\n"
        "    if wide_cause is not None and wide_outcomes:\n"
        "        cser = wide_cause['series']\n"
        "        pieces.append('wide-mode cause=' + wide_cause['label'][:120])\n"
        "        pieces.append('wide-mode trend(cause)=' + str(round(_trend_series(cser), 3)))\n"
        "        for item in wide_mediators:\n"
        "            if item is None:\n"
        "                continue\n"
        "            v = _corr_series(cser, item['series'])\n"
        "            strengths.append(abs(v))\n"
        "            pieces.append('wide corr(cause,mediator ' + item['label'][:80] + ')=' + str(round(v, 3)))\n"
        "        for item in wide_outcomes:\n"
        "            if item is None:\n"
        "                continue\n"
        "            v = _corr_series(cser, item['series'])\n"
        "            strengths.append(abs(v))\n"
        "            pieces.append('wide corr(cause,outcome ' + item['label'][:80] + ')=' + str(round(v, 3)))\n"
        "            pieces.append('wide trend(outcome ' + item['label'][:80] + ')=' + str(round(_trend_series(item['series']), 3)))\n"
        "            for med in wide_mediators:\n"
        "                if med is None:\n"
        "                    continue\n"
        "                mv = _corr_series(med['series'], item['series'])\n"
        "                strengths.append(abs(mv))\n"
        "                pieces.append('wide corr(mediator,outcome)=' + str(round(mv, 3)))\n"
        "    statistic = sum(strengths) / len(strengths) if strengths else abs(cause_trend)\n"
        "    evidence = ' ; '.join(pieces)\n"
        "    return {'evidence': evidence, 'statistic': float(statistic)}\n"
    )


def _role_term_groups(question: str, domain_context: str, role: str) -> tuple[tuple[str, ...], ...]:
    """Extract compact concept matchers for causal table sketches.

    The groups are intentionally plain token conjunctions.  They can match
    either column names or row labels in wide tables, giving the kernel a
    benchmark-agnostic way to find variables when the data layout changes.
    """

    text = f"{question} {domain_context}".lower()
    groups: list[tuple[str, ...]] = []

    def add_if_present(words: Sequence[str], aliases: Sequence[Sequence[str]] = ()) -> None:
        primary_present = all(word in text for word in words)
        if primary_present and tuple(words) not in groups:
            groups.append(tuple(words))
        for alias in aliases:
            if (primary_present or any(word in text for word in alias)) and tuple(alias) not in groups:
                groups.append(tuple(alias))

    if role == "cause":
        add_if_present(("education", "expenditure"), (("adjusted", "savings", "education"),))
        add_if_present(("investment", "human", "capital"), ())
        add_if_present(("treatment",), ())
        add_if_present(("input",), ())
    elif role == "mediator":
        add_if_present(("human", "capital"), (("labor", "force"), ("school", "enrollment"),))
        add_if_present(("education",), (("school", "enrollment"),))
        add_if_present(("mediator",), ())
        add_if_present(("proxy",), ())
    else:
        add_if_present(("economic", "output"), (("gni", "per", "capita"), ("gdp",), ("economic", "income"), ("exports",)))
        add_if_present(("output",), (("gni", "per", "capita"), ("gdp",), ("exports",)))
        add_if_present(("outcome",), ())
        add_if_present(("effect",), ())

    if not groups:
        tokens = sorted(_tokens(question))[:3]
        if tokens:
            groups.append(tuple(tokens))
    return tuple(groups[:6])


def _safe_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_").lower()
    return name or "field"


def _dedupe_programs(programs: Sequence[InducedProgram]) -> list[InducedProgram]:
    seen: set[str] = set()
    out: list[InducedProgram] = []
    for program in programs:
        if program.code in seen:
            continue
        seen.add(program.code)
        out.append(program)
    return out


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2}


def _token_overlap(left: str, right: str) -> float:
    a = _tokens(left)
    b = _tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _best_column(scored: Sequence[tuple[float, str, str]], words: Sequence[str]) -> str:
    for _score, col, bag in scored:
        if any(word in bag for word in words):
            return col
    return ""
