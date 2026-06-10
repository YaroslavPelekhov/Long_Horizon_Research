"""UltraHorizon Grid as a CompressionTask. Same engine, same principle.

A letter's hidden effect is the shortest program mapping the state at the moment
of stepping (x, y, energy, steps, visit_count) to the score change. Execution is
aligned with correctness: the program either reproduces the observed Δscore or
not. One CompressionTask per letter (A-E), like the 5 rule slots in Seq.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

from mars.mdl.engine import CompressionTask


@dataclass
class GridObs:
    x: int
    y: int
    energy: int          # energy at the moment the effect fires (after move cost)
    steps: int           # step count at the moment of the effect
    visit_count: int     # how many times this letter has been visited (incl. this)
    delta_score: int     # observed score change attributable to the letter


class GridLetterTask(CompressionTask):
    entry_point = "f"

    def __init__(self, letter: str, obs: list[GridObs]):
        self.name = f"grid_letter_{letter}"
        self.letter = letter
        self._obs = obs

    def observations(self) -> list[GridObs]:
        return self._obs

    def signature(self) -> str:
        return ("def f(context):\n"
                "    x = context['x']; y = context['y']; energy = context['energy']\n"
                "    steps = context['steps']; visit_count = context['visit_count']\n"
                "    return 0   # replace with the integer score change")

    def interface_hint(self) -> str:
        return ("context is a DICT — access fields as context['x'], context['energy'], etc. "
                "(do NOT unpack it like a tuple). Write VALID, readable Python; correctness "
                "matters far more than character count. Grid is 10x10 (0..9). The effect may "
                "depend on position (parity (x+y)%2, corners/edges, x-y), energy thresholds "
                "(e.g. energy>=15), step modulo (steps%3), or visit_count. Return an int "
                "(may be negative or 0). math is available.")

    def run(self, fn: Callable, obs: GridObs) -> Any:
        return fn({"x": obs.x, "y": obs.y, "energy": obs.energy,
                   "steps": obs.steps, "visit_count": obs.visit_count})

    def target(self, obs: GridObs) -> Any:
        return obs.delta_score

    def matches(self, prediction: Any, obs: GridObs) -> bool:
        try:
            return int(prediction) == int(obs.delta_score)
        except Exception:
            return False

    def per_item_bits(self) -> float:
        # cost to encode one integer score change verbatim
        vals = [abs(o.delta_score) for o in self._obs]
        rng = max(vals) if vals else 1
        return math.log2(2 * rng + 2)  # signed small integer

    def render_observation(self, obs: GridObs) -> dict:
        return {"input": {"x": obs.x, "y": obs.y, "energy": obs.energy,
                          "steps": obs.steps, "visit_count": obs.visit_count},
                "output": obs.delta_score}
