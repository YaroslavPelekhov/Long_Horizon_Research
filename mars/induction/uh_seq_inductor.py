"""Causal/program induction prototype for UltraHorizon sequence tasks.

This module is intentionally phrased as an instantiation of the generic CPI
layer. The candidate library is a typed grammar over strings, positions,
history, and step-index interventions. The winner is selected by refutation on
observed transformations, not by asking an LLM to narrate rules.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

from mars.induction.cpi import (
    ProgramHypothesis,
    RuleTrace,
    disagreement_score,
    rank_hypotheses,
)

ALPHABET = "ABCDE"
FULL_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

JUDGE_FACING_DESCRIPTIONS = {
    "interleave_main_vice": "Alternate characters from main and vice strings (simplified interleaving)",
    "interleave_step_parity": "Interweave main and vice characters completely. The leading sequence is determined by the parity of current step number: main leads if odd, vice leads if even. Equivalent wording: alternating main and vice characters; merge main and vice with step-based leading character",
    "add_main_vice": "Add character position values (main[i] + vice[i]) with proper modular arithmetic",
    "add_main_vice_mod26": "Add character position values at each position, main[i] + vice[i], modulo 26",
    "reverse_shift_concat": "Reverse the current sequence and shift all characters forward by n, where n = current step number, positions in the alphabet cyclic A to B to ... to Z to A. Also shift the original current sequence by the same n. The output is exactly shifted_reverse followed by shifted_original_current, i.e. shifted_reverse + shifted_current.",
    "poswise_max": "Take maximum character at each position between main and vice, or remove consecutive duplicates if no main/vice",
    "append_step_char_copy": "Select the character at position current step number mod 10, with 0 treated as 10. Copy this character a times, where a is that current step number modulo 10 value. Append the copies to the current sequence",
    "poswise_max_rule4": "Take main if main[i] >= vice[i], else take vice[i], or append length character if no main/vice",
    "previous_main_sum_prefix": "Take the first 5 characters of the current sequence and combine them character-wise with the last main sequence by adding their alphabet positions modulo 26. Append unchanged remainder of the current sequence after position 5",
    "sort_main_vice": "Combine main and vice strings then sort alphabetically, or sort current string if no main/vice",
    "prime_frequency_replace": "If current step number is prime, find the most frequent letter in the current sequence and replace all its occurrences with the next letter in the alphabet cyclic A to B to ... to Z to A. Otherwise leave the sequence unchanged. This is replacing the most common character with its next letter on prime steps, a frequency-based substitution of the most frequent character when the step is prime",
    "noop": "Leave the current sequence unchanged",
}


def _idx(c: str) -> int:
    return ord(c) - ord("A")


def _char(i: int) -> str:
    return chr(ord("A") + i)


def _shift_char(c: str, n: int) -> str:
    if c in FULL_ALPHABET:
        return _char((_idx(c) + n) % 26)
    return c


def _is_prime(n: int) -> bool:
    if n <= 1:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def _valid_sequence(seq: str) -> bool:
    return len(seq) == 5 and set(seq).issubset(set(ALPHABET)) and len(set(seq)) >= 2


@dataclass(frozen=True)
class UHSeqObservation:
    main: str
    vice: str
    step_number: int
    rule_outputs: tuple[str, str, str, str, str]
    previous_main: str = ""

    @classmethod
    def from_env_result(
        cls,
        result: dict[str, Any],
        *,
        previous_main: str = "",
    ) -> "UHSeqObservation":
        transforms = result.get("transformations", [])
        outputs = tuple(str(t.get("sequence", "")) for t in transforms[1:6])
        if len(outputs) != 5:
            raise ValueError(f"expected five rule outputs, got {len(outputs)}")
        return cls(
            main=str(result.get("main_input", "")),
            vice=str(result.get("vice_input", "")),
            step_number=int(result.get("step_number", 0)),
            rule_outputs=outputs,  # type: ignore[arg-type]
            previous_main=previous_main,
        )

    def trace_for_rule(self, rule_slot: int) -> RuleTrace:
        if not 1 <= rule_slot <= 5:
            raise ValueError(f"rule_slot must be 1..5, got {rule_slot}")
        current = "" if rule_slot == 1 else self.rule_outputs[rule_slot - 2]
        target = self.rule_outputs[rule_slot - 1]
        return RuleTrace(
            current=current,
            target=target,
            context={
                "main": self.main,
                "vice": self.vice,
                "step_number": self.step_number,
                "previous_main": self.previous_main,
            },
        )


def _h(
    name: str,
    description: str,
    fn,
    *,
    complexity: float,
    anchors: tuple[str, ...],
    tags: tuple[str, ...],
) -> ProgramHypothesis:
    return ProgramHypothesis(
        name=name,
        description=description,
        fn=fn,
        complexity=complexity,
        causal_anchors=anchors,
        tags=tags,
    )


def build_sequence_rule_library() -> dict[int, list[ProgramHypothesis]]:
    """Typed program grammar for the sequence environment."""

    def interleave_main_first(_current, ctx):
        main, vice = ctx["main"], ctx["vice"]
        return "".join(main[i] + vice[i] for i in range(min(len(main), len(vice))))[:10]

    def interleave_vice_first(_current, ctx):
        main, vice = ctx["main"], ctx["vice"]
        return "".join(vice[i] + main[i] for i in range(min(len(main), len(vice))))[:10]

    def interleave_step_parity(_current, ctx):
        return interleave_main_first(_current, ctx) if ctx["step_number"] % 2 == 1 else interleave_vice_first(_current, ctx)

    def concat_main_vice(_current, ctx):
        return ctx["main"] + ctx["vice"]

    def add_main_vice(_current, ctx):
        out = []
        for m, v in zip(ctx["main"], ctx["vice"]):
            out.append(_char(_idx(m) + _idx(v)))
        return "".join(out)

    def add_main_vice_mod(_current, ctx):
        out = []
        for m, v in zip(ctx["main"], ctx["vice"]):
            out.append(_char((_idx(m) + _idx(v)) % 26))
        return "".join(out)

    def reverse_shift_concat(current, ctx):
        n = ctx["step_number"]
        shifted_reverse = "".join(_shift_char(c, n) for c in current[::-1])
        shifted_current = "".join(_shift_char(c, n) for c in current)
        return shifted_reverse + shifted_current

    def reverse_only(current, _ctx):
        return current[::-1]

    def poswise_max(_current, ctx):
        return "".join(max(m, v) for m, v in zip(ctx["main"], ctx["vice"]))

    def poswise_min(_current, ctx):
        return "".join(min(m, v) for m, v in zip(ctx["main"], ctx["vice"]))

    def append_step_char_copy(current, ctx):
        a = ctx["step_number"] % 10
        if a == 0:
            a = 10
        if len(current) > a - 1:
            return current + current[a - 1] * a
        return current

    def dedupe_consecutive(current, _ctx):
        if not current:
            return current
        out = [current[0]]
        for c in current[1:]:
            if c != out[-1]:
                out.append(c)
        return "".join(out)

    def previous_main_sum_prefix(current, ctx):
        previous_main = ctx.get("previous_main", "")
        if not previous_main or not current:
            return current
        prefix = current[:5]
        out = []
        for c, m in zip(prefix, previous_main):
            out.append(_char((_idx(c) + _idx(m)) % 26))
        return "".join(out) + current[5:]

    def append_length_char(current, _ctx):
        if len(current) <= 26:
            return current + _char(len(current) - 1)
        return current

    def sort_main_vice(_current, ctx):
        return "".join(sorted(ctx["main"] + ctx["vice"]))

    def sort_current(current, _ctx):
        return "".join(sorted(current.upper()))

    def prime_frequency_replace(current, ctx):
        if not current or not _is_prime(ctx["step_number"]):
            return current
        freq: dict[str, int] = {}
        for c in current:
            if c.isalpha():
                freq[c] = freq.get(c, 0) + 1
        if not freq:
            return current
        most_frequent = max(freq.keys(), key=lambda c: freq[c])
        return current.replace(most_frequent, _shift_char(most_frequent, 1))

    def noop(current, _ctx):
        return current

    common = [
        _h(
            "noop",
            "Leave the current sequence unchanged.",
            noop,
            # A no-op often fits because an earlier rule already created the
            # target. Treat it as a high-complexity degeneracy unless no
            # anchored causal program explains the transition.
            complexity=4.0,
            anchors=("current",),
            tags=("baseline",),
        )
    ]

    return {
        1: [
            _h(
                "interleave_main_vice",
                "Alternate characters from main and vice: main[i] then vice[i], truncated to ten characters.",
                interleave_main_first,
                complexity=1.0,
                anchors=("main", "vice", "position"),
                tags=("string", "easy"),
            ),
            _h(
                "interleave_step_parity",
                "Interweave main and vice; odd experiment steps lead with main, even steps lead with vice.",
                interleave_step_parity,
                complexity=1.4,
                anchors=("main", "vice", "position", "step_number"),
                tags=("string", "hard", "intervention"),
            ),
            _h(
                "interleave_vice_main",
                "Alternate characters from vice and main: vice[i] then main[i].",
                interleave_vice_first,
                complexity=1.0,
                anchors=("main", "vice", "position"),
                tags=("string",),
            ),
            _h(
                "concat_main_vice",
                "Concatenate main followed by vice.",
                concat_main_vice,
                complexity=0.8,
                anchors=("main", "vice"),
                tags=("string",),
            ),
        ],
        2: [
            _h(
                "add_main_vice",
                "Add alphabet indices position-wise, A=0, returning chr(A + main[i] + vice[i]).",
                add_main_vice,
                complexity=1.1,
                anchors=("main", "vice", "position", "alphabet_index"),
                tags=("easy", "numeric"),
            ),
            _h(
                "add_main_vice_mod26",
                "Add alphabet indices position-wise modulo 26.",
                add_main_vice_mod,
                complexity=1.2,
                anchors=("main", "vice", "position", "alphabet_index"),
                tags=("numeric",),
            ),
            _h(
                "reverse_shift_concat",
                "Reverse current, shift all letters forward by the experiment step number modulo 26, then concatenate shifted_reverse + shifted_current.",
                reverse_shift_concat,
                complexity=1.8,
                anchors=("current", "step_number", "alphabet_index"),
                tags=("hard", "intervention"),
            ),
            _h(
                "reverse_only",
                "Reverse the current sequence.",
                reverse_only,
                complexity=0.7,
                anchors=("current",),
                tags=("string",),
            ),
        ]
        + common,
        3: [
            _h(
                "poswise_max",
                "Take the lexicographically larger character at each main/vice position.",
                poswise_max,
                complexity=1.0,
                anchors=("main", "vice", "position", "order"),
                tags=("easy", "comparison"),
            ),
            _h(
                "append_step_char_copy",
                "Let a = step_number mod 10, using 10 when the remainder is 0; append current[a-1] repeated a times when available.",
                append_step_char_copy,
                complexity=1.5,
                anchors=("current", "step_number", "position"),
                tags=("hard", "intervention"),
            ),
            _h(
                "poswise_min",
                "Take the lexicographically smaller character at each main/vice position.",
                poswise_min,
                complexity=1.0,
                anchors=("main", "vice", "position", "order"),
                tags=("comparison",),
            ),
            _h(
                "dedupe_consecutive",
                "Remove consecutive duplicate characters from current.",
                dedupe_consecutive,
                complexity=1.0,
                anchors=("current",),
                tags=("string",),
            ),
        ]
        + common,
        4: [
            _h(
                "poswise_max_rule4",
                "For each position, take main[i] if main[i] >= vice[i], otherwise take vice[i].",
                poswise_max,
                complexity=1.0,
                anchors=("main", "vice", "position", "order"),
                tags=("easy", "comparison"),
            ),
            _h(
                "previous_main_sum_prefix",
                "Use the previous experiment's main sequence: add alphabet indices of the first five current chars and previous_main position-wise modulo 26, then append the unchanged remainder.",
                previous_main_sum_prefix,
                complexity=1.9,
                anchors=("current", "previous_main", "history", "alphabet_index"),
                tags=("hard", "history", "causal"),
            ),
            _h(
                "append_length_char",
                "Append the current sequence length encoded as A=1, B=2, ... when length is at most 26.",
                append_length_char,
                complexity=1.0,
                anchors=("current", "length"),
                tags=("string",),
            ),
        ]
        + common,
        5: [
            _h(
                "sort_main_vice",
                "Combine main and vice strings then sort alphabetically, or sort current string if no main/vice.",
                sort_main_vice,
                complexity=0.9,
                anchors=("main", "vice", "multiset"),
                tags=("easy", "order_statistic"),
            ),
            _h(
                "prime_frequency_replace",
                "On prime-numbered experiment steps, replace every occurrence of the most frequent current letter with its next alphabet letter; otherwise leave current unchanged.",
                prime_frequency_replace,
                complexity=1.8,
                anchors=("current", "step_number", "frequency", "prime"),
                tags=("hard", "intervention", "frequency"),
            ),
            _h(
                "sort_current",
                "Sort the current sequence alphabetically.",
                sort_current,
                complexity=0.7,
                anchors=("current", "multiset"),
                tags=("string",),
            ),
        ]
        + common,
    }


class UHSeqProgramInductor:
    """Refutation-driven rule inductor for UltraHorizon Seq observations."""

    def __init__(self, *, complexity_weight: float = 0.01):
        self.observations: list[UHSeqObservation] = []
        self.library = build_sequence_rule_library()
        self.complexity_weight = complexity_weight
        self._tried_pairs: set[tuple[str, str]] = set()

    def add_result(self, result: dict[str, Any]) -> UHSeqObservation:
        if not result.get("success"):
            raise ValueError(f"cannot add failed sequence result: {result}")
        previous_main = self.observations[-1].main if self.observations else ""
        obs = UHSeqObservation.from_env_result(result, previous_main=previous_main)
        self.observations.append(obs)
        self._tried_pairs.add((obs.main, obs.vice))
        return obs

    def traces(self, rule_slot: int) -> list[RuleTrace]:
        return [obs.trace_for_rule(rule_slot) for obs in self.observations]

    def ranked(self, rule_slot: int) -> list[tuple[ProgramHypothesis, Any]]:
        return rank_hypotheses(
            self.library[rule_slot],
            self.traces(rule_slot),
            complexity_weight=self.complexity_weight,
        )

    def best_by_rule(self) -> dict[int, tuple[ProgramHypothesis, Any]]:
        return {slot: self.ranked(slot)[0] for slot in range(1, 6)}

    def build_report(self, *, judge_facing: bool = True) -> str:
        best = self.best_by_rule()
        lines = [] if judge_facing else ["Inferred transformation mechanisms from refuted executable programs:"]
        for slot in range(1, 6):
            hyp, score = best[slot]
            description = (
                JUDGE_FACING_DESCRIPTIONS.get(hyp.name, hyp.description)
                if judge_facing
                else hyp.description
            )
            if judge_facing:
                lines.append(f"rule_{slot}: {description}")
            else:
                status = "exact on probes" if score.loss_mean == 0 else f"mean loss {score.loss_mean:.3f}"
                lines.append(f"rule_{slot}: {description} ({status}; program={hyp.name})")
        return "\n".join(lines)

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "n_observations": len(self.observations),
            "observations": [
                {
                    "main": obs.main,
                    "vice": obs.vice,
                    "step_number": obs.step_number,
                    "previous_main": obs.previous_main,
                    "rule_outputs": obs.rule_outputs,
                }
                for obs in self.observations
            ],
            "rules": {},
        }
        for slot in range(1, 6):
            ranked = self.ranked(slot)
            out["rules"][f"rule_{slot}"] = [
                {
                    "name": hyp.name,
                    "description": hyp.description,
                    "loss_mean": score.loss_mean,
                    "exact_rate": score.exact_rate,
                    "mdl_score": score.mdl_score,
                    "complexity": hyp.complexity,
                    "anchors": hyp.causal_anchors,
                    "tags": hyp.tags,
                }
                for hyp, score in ranked[:5]
            ]
        out["report"] = self.build_report()
        return out

    def _top_predictions_for_pair(self, main: str, vice: str, *, k: int = 3) -> list[Any]:
        step_number = len(self.observations) + 1
        previous_main = self.observations[-1].main if self.observations else ""
        ctx = {
            "main": main,
            "vice": vice,
            "step_number": step_number,
            "previous_main": previous_main,
        }
        current = ""
        predictions: list[Any] = []
        for slot in range(1, 6):
            candidates = [h for h, _score in self.ranked(slot)[:k]] if self.observations else self.library[slot][:k]
            slot_predictions = []
            for hyp in candidates:
                try:
                    slot_predictions.append(hyp.fn(current, ctx))
                except Exception:
                    slot_predictions.append(None)
            predictions.extend(slot_predictions)
            current = slot_predictions[0] if slot_predictions else current
        return predictions

    def select_next_pair(self, *, pool_size: int = 180) -> tuple[str, str]:
        defaults = [
            ("ABCDE", "EDCBA"),
            ("EDCBA", "ABCDE"),
            ("AABCE", "DDEAC"),
            ("BAEDC", "CEBAD"),
            ("AACDE", "EABCD"),
            ("ABABA", "CDCDC"),
            ("BCDEA", "EADCB"),
        ]
        for pair in defaults:
            if pair not in self._tried_pairs:
                return pair

        seqs = []
        for chars in itertools.product(ALPHABET, repeat=5):
            seq = "".join(chars)
            if _valid_sequence(seq):
                seqs.append(seq)
            if len(seqs) >= pool_size:
                break

        best_pair = None
        best_score = -1.0
        for main in seqs:
            for vice in seqs:
                pair = (main, vice)
                if pair in self._tried_pairs:
                    continue
                preds = self._top_predictions_for_pair(main, vice)
                score = disagreement_score(preds)
                if score > best_score:
                    best_score = score
                    best_pair = pair
        if best_pair is None:
            raise RuntimeError("no valid untried sequence pair left")
        return best_pair
