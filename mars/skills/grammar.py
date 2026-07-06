"""General self-compressing hypothesis grammar.

The novelty target here is deliberately not benchmark code.  A benchmark may
provide observations and a verifier, but this layer only knows how to:

1. represent a skill as a falsifiable hypothesis-constructor contract;
2. birth a local skill from structured residuals;
3. consolidate multiple local skills into a more general program schema by
   anti-unification under an MDL-style compression criterion.

This gives MARS a stronger object than a prompt or a textual memory item:
skills are small typed grammar rules that generate, verify, repair, and compress
hypotheses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


Expr = str | int | float | tuple["Expr", ...]


@dataclass(frozen=True)
class ResidualCase:
    """One counterexample produced by a failed hypothesis.

    Features are task-observable variables only.  The layer does not need to
    know which benchmark produced them.
    """

    features: dict[str, Any]
    prediction: Any
    target: Any
    loss: float
    context: dict[str, Any] = field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return self.loss > 0.0


@dataclass(frozen=True)
class SkillContract:
    """A reusable, falsifiable hypothesis-constructor.

    `schema` is a tiny program grammar expression.  Concrete examples:

      ("threshold_split", "energy", "<", 15)
      ("interaction_product", "mass1", "mass2")

    A consolidated skill may contain variables:

      ("threshold_split", "?v0", "<", "?v1")

    The contract is intentionally textual/structural rather than bound to one
    Python implementation, so benchmark adapters can compile it into their own
    candidate programs.
    """

    name: str
    schema: Expr
    detector: str
    generator: str
    verifier: str
    repair: str
    compression_cost: float
    support: int = 1
    examples: tuple[str, ...] = ()
    parents: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    @property
    def structural_key(self) -> str:
        return expr_to_text(self.schema)


@dataclass(frozen=True)
class AntiUnification:
    """Least-general generalization result for a set of skill schemas."""

    schema: Expr
    variables: tuple[str, ...]
    bindings: tuple[dict[str, Expr], ...]
    separate_cost: float
    generalized_cost: float

    @property
    def compression_gain(self) -> float:
        return self.separate_cost - self.generalized_cost


@dataclass(frozen=True)
class SkillInductionReport:
    born: tuple[SkillContract, ...]
    consolidated: tuple[SkillContract, ...]
    rejected: tuple[str, ...]


def expr_to_text(expr: Expr) -> str:
    if isinstance(expr, tuple):
        return "(" + " ".join(expr_to_text(x) for x in expr) + ")"
    return str(expr)


def expr_cost(expr: Expr) -> float:
    """Small MDL proxy: constants and atoms cost bits; variables are cheap."""
    if isinstance(expr, tuple):
        return 1.0 + sum(expr_cost(x) for x in expr)
    if isinstance(expr, str) and expr.startswith("?"):
        return 0.35
    if isinstance(expr, (int, float)):
        return 1.4
    return 1.0


def _anti_unify_pair(left: Expr, right: Expr, var_counter: list[int]) -> tuple[Expr, dict[str, Expr], dict[str, Expr]]:
    if left == right:
        return left, {}, {}
    if isinstance(left, tuple) and isinstance(right, tuple) and len(left) == len(right) and left[:1] == right[:1]:
        out: list[Expr] = [left[0]]
        lb: dict[str, Expr] = {}
        rb: dict[str, Expr] = {}
        for l_child, r_child in zip(left[1:], right[1:]):
            child, child_lb, child_rb = _anti_unify_pair(l_child, r_child, var_counter)
            out.append(child)
            lb.update(child_lb)
            rb.update(child_rb)
        return tuple(out), lb, rb
    name = f"?v{var_counter[0]}"
    var_counter[0] += 1
    return name, {name: left}, {name: right}


def _match_template(template: Expr, concrete: Expr, bindings: dict[str, Expr] | None = None) -> dict[str, Expr]:
    bindings = {} if bindings is None else dict(bindings)
    if isinstance(template, str) and template.startswith("?"):
        existing = bindings.get(template)
        if existing is not None and existing != concrete:
            raise ValueError(f"inconsistent binding for {template}: {existing!r} vs {concrete!r}")
        bindings[template] = concrete
        return bindings
    if isinstance(template, tuple) and isinstance(concrete, tuple) and len(template) == len(concrete):
        for t_child, c_child in zip(template, concrete):
            bindings = _match_template(t_child, c_child, bindings)
        return bindings
    if template != concrete:
        raise ValueError(f"schema mismatch: {template!r} vs {concrete!r}")
    return bindings


def _normalize_variables(expr: Expr) -> Expr:
    mapping: dict[str, str] = {}

    def walk(item: Expr) -> Expr:
        if isinstance(item, str) and item.startswith("?"):
            if item not in mapping:
                mapping[item] = f"?v{len(mapping)}"
            return mapping[item]
        if isinstance(item, tuple):
            return tuple(walk(x) for x in item)
        return item

    return walk(expr)


def anti_unify(schemas: Iterable[Expr], *, parameter_weight: float = 0.42) -> AntiUnification:
    """Compute a compact generalization of several program schemas.

    This is not semantic theorem proving.  It is a small, deterministic
    least-general-generalization operator that turns repeated local skills into
    parameterized parent skills when their program skeleton matches.
    """

    items = tuple(schemas)
    if not items:
        raise ValueError("anti_unify requires at least one schema")
    if len(items) == 1:
        schema = items[0]
        return AntiUnification(
            schema=schema,
            variables=(),
            bindings=({},),
            separate_cost=expr_cost(schema),
            generalized_cost=expr_cost(schema),
        )

    var_counter = [0]
    schema = items[0]
    for item in items[1:]:
        schema, _, _ = _anti_unify_pair(schema, item, var_counter)
    schema = _normalize_variables(schema)
    bindings = tuple(_match_template(schema, item) for item in items)
    variables = tuple(sorted({v for b in bindings for v in b}))
    separate = sum(expr_cost(item) for item in items)
    generalized = expr_cost(schema) + parameter_weight * sum(expr_cost(v) for b in bindings for v in b.values())
    return AntiUnification(
        schema=schema,
        variables=variables,
        bindings=bindings,
        separate_cost=separate,
        generalized_cost=generalized,
    )


def _best_numeric_threshold(cases: tuple[ResidualCase, ...]) -> tuple[str, float, str, float] | None:
    failed = [c for c in cases if c.failed]
    passed = [c for c in cases if not c.failed]
    if len(failed) < 2 or not passed:
        return None
    best: tuple[str, float, str, float] | None = None
    keys = sorted({k for c in cases for k, v in c.features.items() if isinstance(v, (int, float))})
    for key in keys:
        values = sorted({float(c.features[key]) for c in cases if key in c.features})
        for left, right in zip(values, values[1:]):
            threshold = (left + right) / 2.0
            for op in ("<=", ">"):
                if op == "<=":
                    selected = [c for c in cases if float(c.features.get(key, float("inf"))) <= threshold]
                else:
                    selected = [c for c in cases if float(c.features.get(key, float("-inf"))) > threshold]
                if not selected:
                    continue
                precision = sum(c.failed for c in selected) / len(selected)
                recall = sum(c.failed for c in selected) / len(failed)
                score = 2 * precision * recall / max(precision + recall, 1e-9)
                if best is None or score > best[3]:
                    best = (key, threshold, op, score)
    if best and best[3] >= 0.66:
        return best
    return None


def birth_skill_from_residuals(name: str, cases: Iterable[ResidualCase]) -> SkillContract | None:
    """Create a local skill if residuals have a compressed explanation."""

    items = tuple(cases)
    threshold = _best_numeric_threshold(items)
    if threshold is not None:
        key, value, op, score = threshold
        rounded = round(value, 4)
        return SkillContract(
            name=name,
            schema=("threshold_split", key, op, rounded),
            detector=f"residuals cluster where {key} {op} {rounded} (F1={score:.2f})",
            generator=f"generate piecewise hypotheses conditioned on {key} {op} T",
            verifier="compare each branch on held-out residuals",
            repair="move threshold, add branch, or reject if holdout residual rises",
            compression_cost=expr_cost(("threshold_split", key, op, rounded)),
            support=sum(c.failed for c in items),
            examples=tuple(str(c.features) for c in items if c.failed)[:3],
            tags=("failure-born", "threshold"),
        )

    pair_keys = sorted(
        {
            k
            for c in items
            for k, v in c.features.items()
            if isinstance(v, (int, float))
            and len({other.features.get(k) for other in items if k in other.features}) > 1
        }
    )
    if len(pair_keys) >= 2 and sum(c.failed for c in items) >= 2:
        left, right = pair_keys[:2]
        return SkillContract(
            name=name,
            schema=("interaction_product", left, right),
            detector=f"residuals vary with a pair of numeric factors: {left}, {right}",
            generator=f"generate hypotheses using product/interactions of {left} and {right}",
            verifier="ablate each factor and test holdout residual",
            repair="swap pair, add exponent, or reject if interaction is not predictive",
            compression_cost=expr_cost(("interaction_product", left, right)),
            support=sum(c.failed for c in items),
            examples=tuple(str(c.features) for c in items if c.failed)[:3],
            tags=("failure-born", "interaction"),
        )
    return None


class SkillGrammar:
    """Library that grows by residuals and shrinks by MDL consolidation."""

    def __init__(self, *, min_gain: float = 0.5) -> None:
        self.min_gain = min_gain
        self.skills: list[SkillContract] = []
        self.rejected: list[str] = []

    def add(self, skill: SkillContract) -> None:
        if skill.structural_key not in {s.structural_key for s in self.skills}:
            self.skills.append(skill)

    def birth(self, name: str, cases: Iterable[ResidualCase]) -> SkillContract | None:
        skill = birth_skill_from_residuals(name, cases)
        if skill is None:
            self.rejected.append(f"{name}: residuals did not compress into a supported skill")
            return None
        self.add(skill)
        return skill

    def consolidate(self) -> tuple[SkillContract, ...]:
        grouped: dict[str, list[SkillContract]] = {}
        for skill in self.skills:
            if isinstance(skill.schema, tuple) and skill.schema:
                grouped.setdefault(str(skill.schema[0]), []).append(skill)

        parents: list[SkillContract] = []
        for head, group in grouped.items():
            if len(group) < 2:
                continue
            au = anti_unify([s.schema for s in group])
            if au.compression_gain < self.min_gain:
                self.rejected.append(
                    f"{head}: anti-unification gain {au.compression_gain:.2f} below {self.min_gain:.2f}"
                )
                continue
            parent = SkillContract(
                name=f"general_{head}",
                schema=au.schema,
                detector=f"select a concrete {head} instantiation by matching task observables",
                generator=f"instantiate {expr_to_text(au.schema)} and generate candidate hypotheses",
                verifier="validate instantiated child on held-out residuals before reuse",
                repair="specialize variables or split the parent if compression no longer preserves score",
                compression_cost=au.generalized_cost,
                support=sum(s.support for s in group),
                examples=tuple(s.name for s in group),
                parents=tuple(s.name for s in group),
                tags=("consolidated", "anti-unified", "mdl"),
            )
            parents.append(parent)
            self.add(parent)
        return tuple(parents)

    def induce(self, batches: Iterable[tuple[str, Iterable[ResidualCase]]]) -> SkillInductionReport:
        born: list[SkillContract] = []
        for name, cases in batches:
            skill = self.birth(name, cases)
            if skill is not None:
                born.append(skill)
        consolidated = self.consolidate()
        return SkillInductionReport(
            born=tuple(born),
            consolidated=consolidated,
            rejected=tuple(self.rejected),
        )
