"""NewtonBench as a CompressionTask. Same engine, same single principle.

A law is whatever shortest program reproduces (inputs -> force) within
tolerance. The multiplicative constant is fitted by calibrate() (data-driven),
so the proposer only has to find the STRUCTURE; bits then judge the fit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

from mars.mdl.engine import CompressionTask


@dataclass
class NBObs:
    inputs: dict
    force: float


class NewtonLawTask(CompressionTask):
    entry_point = "f"

    def __init__(self, name: str, params: list[str], obs: list[NBObs],
                 tol: float = 0.05):
        self.name = name
        self.params = params
        self._obs = obs
        self.tol = tol

    def observations(self) -> list[NBObs]:
        return self._obs

    def signature(self) -> str:
        return (f"def f(inputs):  # inputs is a dict with keys {self.params}\n"
                f"    # return a float (the law's output). math is available.")

    def interface_hint(self) -> str:
        return ("There may be an unknown multiplicative constant — it is fitted "
                "automatically, so focus on the STRUCTURE (which variables, powers, "
                "products, ratios). Return the structural expression; a leading "
                "constant of 1.0 is fine.")

    def run(self, fn: Callable, obs: NBObs) -> Any:
        k = getattr(fn, "_k", 1.0)
        return k * float(fn(obs.inputs))

    def target(self, obs: NBObs) -> Any:
        return obs.force

    def matches(self, prediction: Any, obs: NBObs) -> bool:
        try:
            p = float(prediction); t = float(obs.force)
        except Exception:
            return False
        if p != p or t != t:
            return False
        return abs(p - t) <= self.tol * (abs(t) + 1e-12)

    def item_residual_bits(self, prediction: Any, obs: NBObs) -> float | None:
        """Graded residual: bits to fix the prediction to the true value at the
        target precision. ~0 when relative error << tol; grows with log error."""
        try:
            p = float(prediction); t = float(obs.force)
        except Exception:
            return self.per_item_bits()
        if p != p or t != t:
            return self.per_item_bits()
        rel = abs(p - t) / (abs(t) + 1e-12)
        if rel <= self.tol:
            return 0.0
        # bits needed to correct an error of size rel (capped at verbatim cost)
        return min(self.per_item_bits(), math.log2(rel / self.tol + 1.0) + 1.0)

    def calibrate(self, fn: Callable, observations: list[NBObs]) -> Callable:
        """Fit one multiplicative constant k via geometric mean of target/base."""
        ratios = []
        for o in observations:
            try:
                base = float(fn(o.inputs))
                if base != 0 and base == base and o.force == o.force and o.force != 0:
                    r = o.force / base
                    if r > 0:
                        ratios.append(math.log(r))
            except Exception:
                continue
        k = math.exp(sum(ratios) / len(ratios)) if ratios else 1.0
        try:
            fn._k = k  # type: ignore[attr-defined]
        except Exception:
            pass
        return fn

    def per_item_bits(self) -> float:
        # cost to store one force value verbatim at the matching precision.
        # ~ -log2(tol) bits of mantissa + exponent magnitude.
        mant = -math.log2(self.tol)
        exps = [abs(math.log10(abs(o.force))) for o in self._obs if o.force]
        expo = (sum(exps) / len(exps)) * math.log2(10) if exps else 8.0
        return mant + expo

    def render_observation(self, obs: NBObs) -> dict:
        return {"input": {k: round(v, 5) for k, v in obs.inputs.items()},
                "output": float(f"{obs.force:.5g}")}
