"""Popperian CPI — hypotheses are PROHIBITIONS, not predictions.

The core inversion
------------------
Every prior automated-discovery system (AI Scientist, FunSearch, AlphaEvolve,
and our own UniversalCPI) defines a hypothesis as a function x -> y and
minimizes prediction loss. They optimize what is OBSERVED.

Popper's insight: a scientific hypothesis is defined by what it FORBIDS. Its
content is the size of the space it rules out. This module makes that executable.

A hypothesis here is a PREDICATE:

    allows(event) -> bool      # True = event is permitted; False = forbidden

Its strength is EMPIRICAL CONTENT = the fraction of the plausible event space
it forbids. A hypothesis is REFUTED the moment one real observation falls in its
forbidden zone. Search maximizes content subject to zero refutations.

This is not minimize-loss. It is maximize-prohibition-at-zero-violations.

Why it matters (concrete)
-------------------------
Our prediction-first Bio system scored 0/20 on the lethal-combination criterion.
Lethality is not a prediction — it is a PROHIBITION on existence ("this genotype
cannot produce viable offspring"). A loss-minimizer is structurally blind to it,
because forbidden events are exactly the ones never observed. A prohibition
engine finds it first, because "combination C never appears among viable
offspring" is a high-content hypothesis that the data does not refute.

Selection criterion
--------------------
    score(h) = empirical_content(h)         if h is not refuted by real data
             = -inf                          if any real observation violates h

Ties broken by minimum description length (simpler prohibition wins).

Experiment selection: SEVERE TESTING (Mayo), not disagreement. Choose the action
whose outcome is most likely to refute a surviving hypothesis regardless of which
way it turns out — the test a hypothesis is least likely to survive by luck.
"""

from __future__ import annotations

import ast
import json
import math
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from mars.agents.base import call_llm, make_openai_client


# ===========================================================================
# Core types
# ===========================================================================

@dataclass
class Prohibition:
    """A hypothesis expressed as a predicate: allows(event) -> bool.

    `allows(event) == False` means the hypothesis FORBIDS that event.
    The prohibition's content is the fraction of plausible events it forbids.
    """
    name: str
    description: str          # human-readable statement of what is FORBIDDEN
    code: str
    fn: Callable             # allows(event) -> bool
    complexity: float = 2.0
    tags: tuple[str, ...] = ("proposed",)

    def allows(self, event: Any) -> bool:
        try:
            return bool(self.fn(event))
        except Exception:
            # A predicate that errors on an event is treated as permitting it
            # (it makes no claim there), so errors never cause false refutation.
            return True


@dataclass
class ProhibitionScore:
    name: str
    content: float           # fraction of plausible space forbidden (0..1)
    refuted: bool
    n_violations: int        # real observations falling in forbidden zone
    mdl_score: float         # content adjusted by complexity (higher = better)
    complexity: float


@dataclass
class PopperResult:
    benchmark: str
    n_real: int
    n_proposed: int
    n_valid: int
    survivors: list[tuple[Prohibition, ProhibitionScore]]   # not refuted, by content
    refuted: list[tuple[Prohibition, ProhibitionScore]]
    theory: str              # conjunction of surviving prohibitions, verbalized
    rounds: int
    errors: list[str]
    wall_time_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "n_real": self.n_real,
            "n_proposed": self.n_proposed,
            "n_valid": self.n_valid,
            "rounds": self.rounds,
            "wall_time_s": self.wall_time_s,
            "survivors": [
                {"name": p.name, "forbids": p.description,
                 "content": round(s.content, 3), "complexity": p.complexity,
                 "code": p.code}
                for p, s in self.survivors
            ],
            "n_refuted": len(self.refuted),
            "errors": self.errors[:10],
        }


# ===========================================================================
# Sandbox (predicate must be pure and safe)
# ===========================================================================

_SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs, "all": all, "any": any, "bool": bool, "chr": chr, "dict": dict,
    "enumerate": enumerate, "float": float, "int": int, "isinstance": isinstance,
    "len": len, "list": list, "max": max, "min": min, "ord": ord, "pow": pow,
    "range": range, "round": round, "set": set, "sorted": sorted, "str": str,
    "sum": sum, "tuple": tuple, "zip": zip, "abs": abs,
    "True": True, "False": False, "None": None,
}
_BANNED = {"eval", "exec", "open", "compile", "__import__", "globals", "locals",
           "getattr", "setattr", "delattr", "vars", "input"}
_BANNED_ATTR = {"__class__", "__globals__", "__code__", "__bases__",
                "__subclasses__", "__mro__", "__dict__", "__builtins__"}


def _safe(code: str) -> tuple[bool, str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            mods = ([a.name for a in n.names] if isinstance(n, ast.Import)
                    else [n.module or ""])
            for m in mods:
                if m.split(".")[0] not in ("math",):
                    return False, f"import not allowed: {m}"
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _BANNED:
            return False, f"unsafe call: {n.func.id}"
        if isinstance(n, ast.Attribute) and n.attr in _BANNED_ATTR:
            return False, f"unsafe attribute: {n.attr}"
    return True, ""


def _compile(code: str) -> tuple[bool, str, Any]:
    ok, err = _safe(code)
    if not ok:
        return False, err, None
    ns: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS, "math": math}
    try:
        exec(compile(code, "<popper>", "exec"), ns)
    except Exception as e:
        return False, f"ExecError: {e}", None
    fn = next((v for k, v in ns.items()
               if callable(v) and not isinstance(v, type)
               and not k.startswith("_") and k != "math"), None)
    if fn is None:
        return False, "no predicate found", None
    return True, "", fn


# ===========================================================================
# Adapter — benchmark binding. NO answers; only the event space.
# ===========================================================================

class PopperAdapter(ABC):
    name = "abstract"

    @abstractmethod
    def interface_description(self) -> str:
        """What an event looks like. Types only, no answers."""

    @abstractmethod
    def predicate_signature(self) -> str:
        """Exact signature of allows(event) -> bool."""

    @abstractmethod
    def real_events(self) -> list[Any]:
        """Actual observations from the environment (for refutation)."""

    @abstractmethod
    def plausible_events(self, n: int) -> list[Any]:
        """Sample n PLAUSIBLE events (observed-value combinations) for measuring
        empirical content. Content is the fraction of THESE a prohibition forbids.
        Must be grounded in observed feature values so trivial prohibitions
        (forbidding impossible events) get no free content."""

    @abstractmethod
    def verbalize(self, survivors: list[tuple[Prohibition, ProhibitionScore]]) -> str:
        """Turn surviving prohibitions into the benchmark's expected report."""

    def event_summary(self, events: list[Any], k: int = 8) -> str:
        """Default: JSON of first k events for the LLM prompt."""
        return json.dumps([_trunc(e) for e in events[:k]], default=str, ensure_ascii=False)[:3000]

    # ----- Optional: ACTIVE severe testing --------------------------------
    # The engine designs strategy; the adapter only exposes ATOMIC actions and
    # executes them. No domain strategy lives in the adapter (OS/driver split).

    def supports_active(self) -> bool:
        return False

    def action_space(self) -> str:
        """Describe the ATOMIC actions of this environment and the current
        objects available to act on (e.g. organism ids + phenotypes). The engine
        composes these atoms into a multi-step plan; the adapter never plans.
        Return a short spec + a JSON action schema."""
        return ""

    def execute_action(self, action: dict) -> list[Any]:
        """Execute ONE atomic environment action, return the new events it
        produced. No strategy here — just translate action -> env API -> events."""
        return []


# ===========================================================================
# The Popperian engine
# ===========================================================================

class PopperianCPI:
    def __init__(
        self,
        *,
        model: str = "openai/gpt-4o",
        n_proposals: int = 16,
        content_samples: int = 400,
        complexity_weight: float = 0.02,
        max_rounds: int = 2,
        min_content: float = 0.02,
        temperature: float = 0.85,
    ):
        self.model = model
        self.n_proposals = n_proposals
        self.content_samples = content_samples
        self.complexity_weight = complexity_weight
        self.max_rounds = max_rounds
        self.min_content = min_content
        self.temperature = temperature
        self._client = None

    def _cl(self):
        if self._client is None:
            self._client = make_openai_client()
        return self._client

    # ----- propose prohibitions ------------------------------------------
    def _prompt(self, adapter: PopperAdapter, real: list[Any], prior: str) -> str:
        return f"""{adapter.interface_description()}

PREDICATE SIGNATURE (your functions MUST match exactly):
{adapter.predicate_signature()}

You write FALSIFIABLE PROHIBITIONS. Each function returns True if an event is
PERMITTED and False if your hypothesis FORBIDS it. A good hypothesis forbids a
LARGE region of plausible events while never contradicting the real data below.

Think like Popper: the more an honest law forbids, the more it says. Look for:
- impossibilities (events that NEVER occur — e.g. lethal/forbidden combinations)
- strict orderings stated as bans ("if A present, B never appears")
- conservation/monotonicity stated as bans ("output never contains X")
- boundaries ("value never exceeds / never below a data-grounded threshold")

REAL OBSERVED EVENTS (your prohibition must NEVER forbid any of these):
{adapter.event_summary(real)}
{prior}

Propose {self.n_proposals} diverse, BOLD prohibitions. Bold = forbids a lot.
They will be scored by empirical content (how much they forbid) and discarded
instantly if any real event violates them.

Return ONLY JSON:
{{
  "prohibitions": [
    {{
      "name": "snake_case",
      "description": "what is FORBIDDEN, stated as a ban",
      "complexity": <1-5>,
      "code": "def allows(event):\\n    # return False where FORBIDDEN\\n    ..."
    }}
  ]
}}"""

    def _propose(self, adapter: PopperAdapter, real: list[Any], prior: str):
        raw = call_llm(
            self._cl(), model=self.model,
            system=("You are a Popperian hypothesis engine. You state bold, "
                    "falsifiable prohibitions as predicates. Return only JSON."),
            user=self._prompt(adapter, real, prior),
            max_tokens=4500, temperature=self.temperature,
        )
        items = _parse(raw)
        probs: list[Prohibition] = []
        errors: list[str] = []
        for it in items:
            if not isinstance(it, dict):
                continue
            code = (it.get("code") or "").strip()
            if not code:
                errors.append(f"{it.get('name','?')}: no code")
                continue
            ok, err, fn = _compile(code)
            if not ok:
                errors.append(f"{it.get('name','?')}: {err}")
                continue
            probs.append(Prohibition(
                name=it.get("name", f"prob_{len(probs)}"),
                description=it.get("description", ""),
                code=code, fn=fn,
                complexity=float(it.get("complexity", 2.0)),
            ))
        return probs, items, errors

    # ----- score: content + refutation -----------------------------------
    def _score(self, p: Prohibition, real: list[Any], plausible: list[Any]) -> ProhibitionScore:
        # Refutation: does p forbid any REAL event?
        violations = sum(1 for e in real if not p.allows(e))
        refuted = violations > 0
        # Content: fraction of PLAUSIBLE events forbidden
        if plausible:
            forbidden = sum(1 for e in plausible if not p.allows(e))
            content = forbidden / len(plausible)
        else:
            content = 0.0
        # MDL-adjusted content: reward content, penalize complexity
        mdl = content - self.complexity_weight * p.complexity
        return ProhibitionScore(
            name=p.name, content=content, refuted=refuted,
            n_violations=violations, mdl_score=mdl, complexity=p.complexity,
        )

    # ----- main loop ------------------------------------------------------
    def run(self, adapter: PopperAdapter) -> PopperResult:
        t0 = time.time()
        real = adapter.real_events()
        if not real:
            return PopperResult(adapter.name, 0, 0, 0, [], [], "", 0, ["no real events"], 0.0)
        plausible = adapter.plausible_events(self.content_samples)

        all_probs: list[Prohibition] = []
        all_items: list[dict] = []
        all_errors: list[str] = []
        prior = ""
        rounds = 0

        for rnd in range(self.max_rounds):
            rounds = rnd + 1
            probs, items, errors = self._propose(adapter, real, prior)
            all_probs.extend(probs)
            all_items.extend(items)
            all_errors.extend(errors)

            # Score everything so far
            scored = [(p, self._score(p, real, plausible)) for p in all_probs]
            survivors = [(p, s) for p, s in scored
                         if not s.refuted and s.content >= self.min_content]
            survivors.sort(key=lambda x: (-x[1].mdl_score, x[0].complexity))

            # If we already have strong, diverse survivors, stop
            if len(survivors) >= 5 and survivors[0][1].content >= 0.5:
                break

            # Build prior context: what survived, what got refuted, to push for
            # BOLDER prohibitions next round.
            top_surv = "; ".join(f"{p.description} (content {s.content:.2f})"
                                 for p, s in survivors[:4]) or "none yet"
            n_ref = sum(1 for _, s in scored if s.refuted)
            prior = (
                f"\nPREVIOUS ROUND: {len(survivors)} survived, {n_ref} were refuted "
                f"by real data. Strongest survivors so far: {top_surv}. "
                f"Now propose BOLDER prohibitions that forbid even more — especially "
                f"impossibilities (events that never occur in the data) and strict "
                f"ordering bans. Avoid restating survivors; find NEW forbidden regions."
            )

        # ---- ACTIVE SEVERE TESTING -------------------------------------
        # Design experiments that would most severely test surviving
        # prohibitions, run them, and fold the new events back into `real`.
        # This is what lets the engine find impossibilities/lethality that
        # passive analysis cannot: it deliberately tries to CREATE the
        # forbidden event and sees whether reality permits it.
        severe_log: list[dict] = []
        if adapter.supports_active():
            current_survivors = [p for p, s in
                                 sorted([(p, self._score(p, real, plausible)) for p in all_probs],
                                        key=lambda x: -x[1].content)
                                 if not s.refuted and s.content >= self.min_content][:8]
            # Engine designs a multi-step PLAN of atomic actions; adapter only
            # executes each atom. All strategy stays in the engine.
            plan = self._design_action_plan(adapter, current_survivors)
            for action in plan:
                try:
                    new_events = adapter.execute_action(action)
                except Exception as e:
                    severe_log.append({"action": action, "error": str(e)})
                    continue
                if new_events:
                    real.extend(new_events)
                severe_log.append({"action": action, "n_new_events": len(new_events)})
            plausible = adapter.plausible_events(self.content_samples)

        scored_all = [(p, self._score(p, real, plausible)) for p in all_probs]
        survivors = [(p, s) for p, s in scored_all
                     if not s.refuted and s.content >= self.min_content]
        survivors.sort(key=lambda x: (-x[1].mdl_score, x[0].complexity))
        refuted = [(p, s) for p, s in scored_all if s.refuted]

        # De-duplicate survivors by description
        seen: set[str] = set()
        uniq = []
        for p, s in survivors:
            key = p.description.strip().lower()[:60]
            if key not in seen:
                seen.add(key)
                uniq.append((p, s))
        survivors = uniq

        theory = adapter.verbalize(survivors) if survivors else "No surviving prohibitions."

        result = PopperResult(
            benchmark=adapter.name, n_real=len(real),
            n_proposed=len(all_items), n_valid=len(all_probs),
            survivors=survivors[:12], refuted=refuted,
            theory=theory, rounds=rounds, errors=all_errors,
            wall_time_s=time.time() - t0,
        )
        result.severe_log = severe_log  # type: ignore[attr-defined]
        return result

    def _design_action_plan(
        self,
        adapter: PopperAdapter,
        survivors: list[Prohibition],
    ) -> list[dict]:
        """Ask the LLM to compose a multi-step PLAN of ATOMIC actions that would
        most severely test the surviving prohibitions — deliberately trying to
        CREATE a forbidden event. All strategy (sequencing, which objects to
        combine, how to reach an impossible combination) is decided HERE, by the
        engine, not by the adapter. The adapter only exposes atomic actions."""
        if not survivors:
            return []
        bans = "\n".join(f"- {p.description}" for p in survivors[:8])
        prompt = f"""{adapter.interface_description()}

SURVIVING PROHIBITIONS (not yet refuted):
{bans}

ENVIRONMENT ACTION SPACE (atomic actions + current objects):
{adapter.action_space()}

Compose a PLAN: an ordered list of atomic actions whose combined effect tries
to CREATE an event some prohibition forbids — the most severe possible test.
You must do the sequencing yourself (e.g. to reach a combination not present in
any single object, first combine two objects, then act on the result). A
prohibition that survives its most severe test becomes a corroborated law.
Prioritize impossibility/lethality claims — those can only be confirmed by
actively attempting the forbidden combination.

Return ONLY JSON: {{"plan": [ {{...atomic action per schema...}}, ... ]}} (max 6 actions)"""
        raw = call_llm(
            self._cl(), model=self.model,
            system="You design multi-step severe-testing plans from atomic actions. Return only JSON.",
            user=prompt, max_tokens=1400, temperature=0.5,
        )
        try:
            obj = json.loads(_extract_json(raw))
            plan = obj.get("plan", []) if isinstance(obj, dict) else []
            return [a for a in plan if isinstance(a, dict)][:6]
        except Exception:
            return []


# ===========================================================================
# Severe testing — choose the experiment most likely to refute a survivor.
# ===========================================================================

def severity(
    candidate_action: Any,
    survivors: list[Prohibition],
    predict_outcomes: Callable[[Any], list[Any]],
) -> float:
    """Expected fraction of survivors refuted by running `candidate_action`.

    `predict_outcomes(action)` returns the plausible outcome events of the action.
    A severe test is one where, whatever the outcome, some surviving hypothesis
    forbids it — so observing the real outcome is likely to kill a survivor.
    """
    if not survivors:
        return 0.0
    outcomes = predict_outcomes(candidate_action)
    if not outcomes:
        return 0.0
    # For each plausible outcome, how many survivors does it refute?
    kill_fractions = []
    for out in outcomes:
        killed = sum(1 for p in survivors if not p.allows(out))
        kill_fractions.append(killed / len(survivors))
    # Severity = average kill power across plausible outcomes
    return sum(kill_fractions) / len(kill_fractions)


# ===========================================================================
# Helpers
# ===========================================================================

def _extract_json(raw: str) -> str:
    if not raw:
        return "{}"
    t = raw.strip()
    if t.startswith("```"):
        lines = t.split("\n")
        t = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).lstrip()
    m = re.search(r'\{.*\}', t, re.DOTALL)
    return m.group() if m else "{}"


def _trunc(v: Any, limit: int = 220) -> Any:
    if isinstance(v, (int, float, bool)) or v is None:
        return v
    s = v if isinstance(v, str) else json.dumps(v, default=str, ensure_ascii=False)
    return s if len(s) <= limit else s[:limit] + "…"


def _parse(raw: str) -> list[dict]:
    if not raw:
        return []
    t = raw.strip()
    if t.startswith("```"):
        lines = t.split("\n")
        t = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:]).lstrip()
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            for k in ("prohibitions", "hypotheses", "programs"):
                if isinstance(obj.get(k), list):
                    return obj[k]
        if isinstance(obj, list):
            return obj
    except Exception:
        pass
    m = re.search(r'\[\s*\{.*\}\s*\]', t, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    m2 = re.search(r'\{.*\}', t, re.DOTALL)
    if m2:
        try:
            obj = json.loads(m2.group())
            for k in ("prohibitions", "hypotheses", "programs"):
                if isinstance(obj.get(k), list):
                    return obj[k]
        except Exception:
            pass
    return []
