"""Compositional rule domain — the clean test of certified-depth accumulation.

A hidden effect is a SUM of primitive contributions:

    delta = parity(ctx) + threshold(ctx) + modulo(ctx) + ...

Each primitive alone is easy (a first-year student solves it). A deep SUM of
3-4 primitives is hard to discover in one shot — the search space of "which
primitives, summed how" is large, so a cold weak model's p(right program) ≈ 0.

But if the weak model has DRILLED each primitive (solved it, certified it by
execution, banked it as a reusable building block), the composite becomes
REACHABLE: it only has to discover the COMBINATION, not re-derive each piece.

This isolates the one variable we care about: does banking certified building
blocks (нарешка) expand what the SAME frozen weak model can reach?

Same CompressionTask contract as Grid/Seq — the universal engine drives it.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable

from mars.mdl.engine import CompressionTask


# ---------------------------------------------------------------------------
# Primitive pool — canonical, parameter-free behaviours.
# Each entry: name -> (python_fn(ctx)->int, reference_source, nl_hint)
# The reference_source is ground truth (used only to GENERATE data, never shown).
# ---------------------------------------------------------------------------

def _parity(c):    return 2 if (c["x"] + c["y"]) % 2 == 0 else 0
def _threshold(c): return 3 if c["energy"] >= 15 else 0
def _modulo(c):    return 1 if c["steps"] % 3 == 0 else 0
def _visit(c):     return 4 if c["visit_count"] <= 2 else 0
def _corner(c):    return 5 if (c["x"] in (0, 9) and c["y"] in (0, 9)) else 0
def _diag(c):      return 2 if c["x"] == c["y"] else 0

PRIMITIVES: dict[str, Callable[[dict], int]] = {
    "parity": _parity,
    "threshold": _threshold,
    "modulo": _modulo,
    "visit": _visit,
    "corner": _corner,
    "diag": _diag,
}

# A short NL nudge per primitive (only the TYPE of dependence, not the answer).
PRIMITIVE_HINTS: dict[str, str] = {
    "parity": "may depend on the parity of (x+y)",
    "threshold": "may depend on an energy threshold",
    "modulo": "may depend on steps modulo a small number",
    "visit": "may depend on visit_count being small",
    "corner": "may depend on being at a grid corner",
    "diag": "may depend on lying on the diagonal x==y",
}


# ---------------------------------------------------------------------------
# Context sampling
# ---------------------------------------------------------------------------

def sample_contexts(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        out.append({
            "x": rng.randint(0, 9),
            "y": rng.randint(0, 9),
            "energy": rng.randint(0, 30),
            "steps": rng.randint(0, 20),
            "visit_count": rng.randint(1, 6),
        })
    return out


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

@dataclass
class CompObs:
    context: dict
    delta: int


# ---------------------------------------------------------------------------
# Task: reproduce delta = f(context) where f is a SUM of primitives.
# ---------------------------------------------------------------------------

class CompositeRuleTask(CompressionTask):
    entry_point = "f"

    def __init__(self, name: str, prim_names: list[str], obs: list[CompObs],
                 mode: str = "composite"):
        self.name = name
        self.prim_names = prim_names          # ground-truth composition (for reporting only)
        self._obs = obs
        self.mode = mode                      # "composite" (sum) or "drill" (single rule)

    # --- construction helpers -------------------------------------------
    @classmethod
    def build(cls, name: str, prim_names: list[str], n_obs: int, seed: int,
              mode: str = "composite") -> "CompositeRuleTask":
        ctxs = sample_contexts(n_obs, seed)
        fns = [PRIMITIVES[p] for p in prim_names]
        obs = [CompObs(c, sum(fn(c) for fn in fns)) for c in ctxs]
        return cls(name, prim_names, obs, mode=mode)

    # --- CompressionTask contract ---------------------------------------
    def observations(self) -> list[CompObs]:
        return self._obs

    def signature(self) -> str:
        return ("def f(context):\n"
                "    x = context['x']; y = context['y']; energy = context['energy']\n"
                "    steps = context['steps']; visit_count = context['visit_count']\n"
                "    return 0   # integer score change")

    def interface_hint(self) -> str:
        if self.mode == "drill":
            return ("context is a DICT — access fields as context['x'], etc. The score "
                    "change follows ONE simple rule depending on a SINGLE aspect: position "
                    "parity (x+y)%2, an energy threshold (e.g. energy>=15), steps modulo a "
                    "small number (e.g. steps%3==0), visit_count being small, being at a grid "
                    "corner, or lying on the diagonal x==y. The output is one fixed integer "
                    "when the condition holds and 0 otherwise. Write VALID readable Python.")
        return ("context is a DICT — access fields as context['x'], etc. The score change "
                "is a SUM of several independent effects, each depending on position "
                "(parity (x+y)%2, corner, diagonal x==y), energy thresholds, steps modulo, "
                "or visit_count. Write VALID readable Python; correctness over brevity. "
                "If LIBRARY building blocks are given, the answer is most likely a SUM of "
                "a few of them, e.g. `return prim_a(context) + prim_b(context)`.")

    def run(self, fn: Callable, obs: CompObs) -> Any:
        return fn(dict(obs.context))

    def target(self, obs: CompObs) -> Any:
        return obs.delta

    def matches(self, prediction: Any, obs: CompObs) -> bool:
        try:
            return int(prediction) == int(obs.delta)
        except Exception:
            return False

    def per_item_bits(self) -> float:
        vals = [abs(o.delta) for o in self._obs]
        rng = max(vals) if vals else 1
        return math.log2(2 * rng + 2)

    def render_observation(self, obs: CompObs) -> dict:
        return {"input": dict(obs.context), "output": obs.delta}


# ---------------------------------------------------------------------------
# Single-primitive drill task (the "нарешка" unit): learn ONE primitive.
# ---------------------------------------------------------------------------

def drill_task(prim_name: str, n_obs: int, seed: int) -> CompositeRuleTask:
    """A task whose hidden rule is a SINGLE primitive — easy, certifiable."""
    return CompositeRuleTask.build(f"drill_{prim_name}", [prim_name], n_obs, seed,
                                   mode="drill")
