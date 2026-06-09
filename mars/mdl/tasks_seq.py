"""UH-Seq as a CompressionTask. No answers — only observations + how to run.

A hidden rule is whatever SHORTEST program reproduces the observed transitions.
Long-horizon flavour: the 5 rules of an episode are solved in sequence, sharing
one Library, so a building block discovered for rule k is free for rule k+1
(transfer = compression across the horizon).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

from mars.mdl.engine import CompressionTask


@dataclass
class SeqObs:
    current: str
    target: str
    main: str
    vice: str
    step_number: int
    previous_main: str


class SeqRuleTask(CompressionTask):
    entry_point = "f"

    def __init__(self, rule_slot: int, obs: list[SeqObs]):
        self.name = f"seq_rule_{rule_slot}"
        self.rule_slot = rule_slot
        self._obs = obs

    def observations(self) -> list[SeqObs]:
        return self._obs

    def signature(self) -> str:
        return ("def f(current, context):  # context: main, vice, step_number, previous_main\n"
                "    # return the output string")

    def interface_hint(self) -> str:
        return ("Strings are uppercase letters. position of c: ord(c)-ord('A'); "
                "char at i: chr(ord('A')+i%26). math is available.")

    def run(self, fn: Callable, obs: SeqObs) -> Any:
        return fn(obs.current, {"main": obs.main, "vice": obs.vice,
                                "step_number": obs.step_number, "previous_main": obs.previous_main})

    def target(self, obs: SeqObs) -> Any:
        return obs.target

    def per_item_bits(self) -> float:
        # cost to store one output string verbatim (alphabet ~26)
        avg_len = sum(len(o.target) for o in self._obs) / max(1, len(self._obs))
        return avg_len * math.log2(26)

    def render_observation(self, obs: SeqObs) -> dict:
        return {
            "input": {"current": obs.current, "context": {
                "main": obs.main, "vice": obs.vice,
                "step_number": obs.step_number, "previous_main": obs.previous_main}},
            "output": obs.target,
        }
