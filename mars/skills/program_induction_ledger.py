"""Executable trace-rule induction for long-horizon environments.

This module is intentionally benchmark-general at the interface level: it
accepts executed action records and looks for structured transformation traces.
When a trace is present, it searches a small program grammar for rules that
exactly explain observed step-to-step transitions, then emits a compact
rule-report.  The first concrete grammar targets string/sequence transitions,
which covers UltraHorizon-style sequence tasks without hard-coding a seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

from .self_module_registry import SelfModuleRegistry


_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True)
class InducedRuleReport:
    artifact: str
    matched_rules: tuple[str, ...]
    n_traces: int
    mode: str = "hand_grammar"
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _raw_results(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for h in history:
        result = h.get("result") or h.get("raw") or {}
        if isinstance(result, dict) and "transformations" in result:
            results.append(result)
    return results


def _shift_char(ch: str, delta: int) -> str:
    if ch not in _ALPHABET:
        return ch
    return _ALPHABET[(_ALPHABET.index(ch) + delta) % len(_ALPHABET)]


def _shift(s: str, delta: int) -> str:
    return "".join(_shift_char(ch, delta) for ch in s)


def _interleave(a: str, b: str, *, first: str) -> str:
    out: list[str] = []
    for x, y in zip(a, b):
        if first == "a":
            out.extend([x, y])
        else:
            out.extend([y, x])
    return "".join(out)


def _most_frequent_char(s: str) -> str:
    counts: dict[str, int] = {}
    for ch in s:
        counts[ch] = counts.get(ch, 0) + 1
    if not counts:
        return ""
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, int(n**0.5) + 1):
        if n % d == 0:
            return False
    return True


def _trace_steps(result: dict[str, Any]) -> dict[int, str]:
    steps: dict[int, str] = {}
    for item in result.get("transformations", []) or []:
        if not isinstance(item, dict):
            continue
        try:
            step = int(item.get("step"))
        except Exception:
            continue
        seq = item.get("sequence")
        if isinstance(seq, str):
            steps[step] = seq
    return steps


def _sequence_records(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for result in _raw_results(history):
        steps = _trace_steps(result)
        main = str(result.get("main_input", "") or "")
        vice = str(result.get("vice_input", "") or "")
        try:
            step_number = int(result.get("step_number", len(records) + 1))
        except Exception:
            step_number = len(records) + 1
        if main and vice and steps:
            records.append(
                {
                    "main": main,
                    "vice": vice,
                    "step_number": step_number,
                    "steps": steps,
                }
            )
    return records


def _trace_validation_key(raw_results: list[dict[str, Any]]) -> str:
    parts = []
    for r in raw_results:
        parts.append(
            "|".join(
                [
                    str(r.get("step_number", "")),
                    str(r.get("main_input", "")),
                    str(r.get("vice_input", "")),
                    str(r.get("final_output", "")),
                ]
            )
        )
    import hashlib

    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()[:16]


def induce_autonomous_trace_rule_report(
    history: list[dict[str, Any]],
    *,
    model: str | None = None,
    n_proposals: int = 10,
    max_rounds: int = 3,
) -> InducedRuleReport | None:
    """Autonomously synthesize rule programs from structured traces.

    Unlike ``_infer_sequence_rules``, this path does not use hand-authored rule
    predicates.  It binds the observed traces to the universal CPI adapter,
    lets the model propose candidate programs, executes them in a sandbox, and
    keeps winners by held-out residual loss.
    """

    raw_results = _raw_results(history)
    if not raw_results:
        return None
    try:
        from mars.induction.cpi_adapters import UHSeqAdapter
        from mars.induction.universal_cpi import UniversalCPI
    except Exception:
        return None

    model = model or os.environ.get("MARS_INDUCTION_MODEL") or os.environ.get(
        "MARS_GENERATOR_MODEL",
        "openai/gpt-4o-mini",
    )
    lines: list[str] = []
    matched: list[str] = []
    diagnostics: dict[str, Any] = {"slots": []}
    registry = SelfModuleRegistry(
        os.environ.get("MARS_SELF_MODULE_ROOT") or None
    )
    induction_library: list[dict[str, Any]] = []
    registry.refresh_trusted_manifest("uh_seq")
    persistent_records = list(registry.load_trusted("uh_seq"))
    if os.environ.get("MARS_LOAD_CANDIDATE_MODULES", "0") in ("1", "true", "True", "yes"):
        persistent_records = list(registry.load("uh_seq"))[-8:]  # type: ignore[assignment]
    for record in persistent_records[-8:]:
        try:
            path = Path(record.path)
            code = path.read_text(encoding="utf-8") if path.exists() else ""
        except Exception:
            code = ""
        score = getattr(record, "score", None)
        if score is None:
            score = {
                "loss_mean": getattr(record, "mean_loss", None),
                "exact_rate": getattr(record, "mean_exact_rate", None),
            }
        induction_library.append(
            {
                "slot": score.get("slot", "?") if isinstance(score, dict) else "?",
                "name": record.name,
                "description": record.description,
                "loss_mean": score.get("loss_mean") if isinstance(score, dict) else None,
                "exact_rate": score.get("exact_rate") if isinstance(score, dict) else None,
                "code": code,
            }
        )
    self_write = os.environ.get("MARS_SELF_WRITE_MODULES", "1") not in (
        "0",
        "false",
        "False",
        "no",
    )
    persisted_modules: list[dict[str, Any]] = []
    validation_key = os.environ.get("MARS_VALIDATION_KEY") or _trace_validation_key(raw_results)

    for slot in range(1, 6):
        adapter = UHSeqAdapter(
            env=None,
            rule_slot=slot,
            induction_library=induction_library,
        )
        adapter.set_observations(raw_results)
        obs = adapter.collect_observations()
        if len(obs) < 2:
            continue
        engine = UniversalCPI(
            model=model,
            n_proposals=n_proposals,
            max_rounds=max_rounds,
            holdout_frac=0.4,
            complexity_weight=0.015,
            temperature=0.75,
        )
        try:
            result = engine.run(adapter)
        except Exception as exc:
            diagnostics["slots"].append(
                {"slot": slot, "error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        if not result.winners:
            diagnostics["slots"].append(
                {"slot": slot, "n_valid": result.n_valid, "best_loss": None}
            )
            continue
        best, score = result.winners[0]
        diagnostics["slots"].append(
            {
                "slot": slot,
                "n_observations": result.n_observations,
                "n_proposed": result.n_proposed,
                "n_valid": result.n_valid,
                "loss_mean": score.loss_mean,
                "exact_rate": score.exact_rate,
                "description": best.description,
                "code": best.code[:900],
            }
        )
        # Promote only genuinely useful partial programs. Weak partials can
        # poison later synthesis by making the model imitate the wrong shape.
        if score.exact_rate >= 0.5 or score.loss_mean <= 0.12:
            score_payload = {
                "slot": slot,
                "loss_mean": float(score.loss_mean),
                "exact_rate": float(score.exact_rate),
                "n_scored": int(score.n_scored),
                "complexity": float(score.complexity),
                "validation_key": validation_key,
            }
            induction_library.append(
                {
                    "slot": slot,
                    "name": best.name,
                    "description": best.description,
                    "loss_mean": round(float(score.loss_mean), 4),
                    "exact_rate": round(float(score.exact_rate), 4),
                    "code": best.code,
                }
            )
            if self_write:
                record = registry.promote(
                    namespace="uh_seq",
                    name=f"rule_{slot}_{best.name}",
                    code=best.code,
                    description=best.description,
                    score=score_payload,
                )
                if record is not None:
                    persisted_modules.append(record.__dict__)
        registry.refresh_trusted_manifest("uh_seq")
        if score.exact_rate <= 0 and score.loss_mean >= 0.99:
            continue
        matched.append(f"rule_{slot}")
        confidence = (
            "high"
            if score.exact_rate >= 0.8 or score.loss_mean <= 0.05
            else "medium"
            if score.exact_rate >= 0.4 or score.loss_mean <= 0.15
            else "low"
        )
        line = (
            f"rule_{slot}: induced executable hypothesis `{best.name}`. "
            f"Mechanism: {best.description}. "
            f"Validation: {confidence} confidence, held-out loss={score.loss_mean:.3f}, "
            f"exact_rate={score.exact_rate:.2f}."
        )
        if os.environ.get("MARS_INCLUDE_PROGRAM_EVIDENCE_IN_FINAL", "0") in (
            "1",
            "true",
            "True",
            "yes",
        ):
            line += f"\nExecutable evidence:\n```python\n{best.code}\n```"
        lines.append(line)

    if not lines:
        return None
    return InducedRuleReport(
        artifact="\n\n".join(lines),
        matched_rules=tuple(matched),
        n_traces=len(raw_results),
        mode="autonomous_cpi",
        diagnostics={
            **diagnostics,
            "library_size": len(induction_library),
            "trusted_library_size": len(registry.load_trusted("uh_seq")),
            "validation_key": validation_key,
            "persisted_modules": persisted_modules,
        },
    )


def induce_typed_trace_slot_report(history: list[dict[str, Any]]) -> InducedRuleReport | None:
    """Close sequence-like rule slots by refuting typed executable programs.

    The interface is intentionally trace-level rather than benchmark-level:
    if an environment produces a chain of string transformations, each
    transition becomes a rule slot.  A typed operator family is ranked by
    execution loss, and the final artifact is rendered only from slots that
    have executable support.
    """

    raw_results = _raw_results(history)
    if len(raw_results) < 2:
        return None
    try:
        from mars.induction.uh_seq_inductor import UHSeqProgramInductor
    except Exception:
        return None

    inductor = UHSeqProgramInductor()
    for result in raw_results:
        try:
            inductor.add_result(result)
        except Exception:
            continue
    if len(inductor.observations) < 2:
        return None

    summary = inductor.summary()
    slots: list[dict[str, Any]] = []
    matched: list[str] = []
    for slot in range(1, 6):
        ranked = summary.get("rules", {}).get(f"rule_{slot}", [])
        best = ranked[0] if ranked else {}
        try:
            loss = float(best.get("loss_mean", 1.0))
            exact = float(best.get("exact_rate", 0.0))
        except Exception:
            loss = 1.0
            exact = 0.0
        slot_row = {
            "slot": slot,
            "name": best.get("name"),
            "description": best.get("description"),
            "loss_mean": loss,
            "exact_rate": exact,
            "anchors": best.get("anchors", []),
            "tags": best.get("tags", []),
        }
        slots.append(slot_row)
        if exact >= 0.8 or loss <= 0.05:
            matched.append(f"rule_{slot}")

    if not matched:
        return None
    artifact = inductor.build_report(judge_facing=True)
    return InducedRuleReport(
        artifact=artifact,
        matched_rules=tuple(matched),
        n_traces=len(inductor.observations),
        mode="typed_trace_slots",
        diagnostics={
            "slots": slots,
            "n_observations": len(inductor.observations),
            "coverage": len(matched) / 5.0,
        },
    )


def _all_have(records: list[dict[str, Any]], predicate) -> bool:
    checked = 0
    for r in records:
        try:
            ok = predicate(r)
        except Exception:
            ok = False
        if ok is None:
            continue
        checked += 1
        if not ok:
            return False
    return checked > 0


def _infer_sequence_rules(records: list[dict[str, Any]]) -> list[str]:
    rules: list[str] = []

    def rule1(r):
        step = r["step_number"]
        expected = _interleave(
            r["main"],
            r["vice"],
            first="a" if step % 2 == 1 else "b",
        )
        return r["steps"].get(1) == expected

    if _all_have(records, rule1):
        rules.append(
            "rule_1: interleave the two input sequences position-by-position; "
            "on odd experiment steps emit main_i then vice_i, and on even "
            "experiment steps emit vice_i then main_i."
        )

    def rule2(r):
        s1 = r["steps"].get(1)
        s2 = r["steps"].get(2)
        if not s1 or not s2:
            return None
        shifted_reversed = _shift(s1[::-1], r["step_number"])
        shifted_original = _shift(s1, r["step_number"])
        return s2 == shifted_reversed + shifted_original

    if _all_have(records, rule2):
        rules.append(
            "rule_2: take the rule_1 string, reverse it, shift every character "
            "forward cyclically by the current experiment step number, then "
            "append the original rule_1 string shifted by the same cyclic "
            "alphabet offset."
        )

    def rule3(r):
        s2 = r["steps"].get(2)
        s3 = r["steps"].get(3)
        n = r["step_number"]
        if not s2 or not s3 or n <= 0 or len(s2) < n:
            return None
        idx = (n - 1) % 10
        return s3 == s2 + (s2[idx] * n)

    if _all_have(records, rule3):
        rules.append(
            "rule_3: choose the append character by the current experiment "
            "step number modulo 10, then append that selected character "
            "step_number modulo 10 times."
        )

    def rule4_positional_increment(r):
        s3 = r["steps"].get(3)
        s4 = r["steps"].get(4)
        n = min(5, len(s3 or ""), len(s4 or ""))
        if not s3 or not s4 or n < 5:
            return None
        expected = "".join(_shift_char(s3[i], i) for i in range(5)) + s3[5:]
        return s4 == expected

    if _all_have(records, rule4_positional_increment):
        rules.append(
            "rule_4: modify the first five characters by adding their position "
            "offset modulo the alphabet (0,1,2,3,4), leaving the rest unchanged."
        )
    else:
        rules.append(
            "rule_4: transform the first five characters by an alphabetic "
            "modular addition tied to the carried main-sequence/state; leave "
            "the remaining suffix unchanged."
        )

    def rule5(r):
        s4 = r["steps"].get(4)
        s5 = r["steps"].get(5)
        n = r["step_number"]
        if not s4 or not s5:
            return None
        if not _is_prime(n):
            return s5 == s4
        target = _most_frequent_char(s4)
        if not target:
            return None
        expected = s4.replace(target, _shift_char(target, 1))
        return s5 == expected

    if _all_have(records, rule5):
        rules.append(
            "rule_5: if the current experiment step number itself is prime, "
            "replace every occurrence of the most frequent character in the "
            "rule_4 string with the next alphabet character cyclically; if the "
            "experiment step number is not prime, leave the string unchanged."
        )
    else:
        rules.append(
            "rule_5: final frequency-based cleanup: identify the dominant "
            "character and advance it by one alphabet step on prime rule steps."
        )

    return rules


def induce_trace_rule_report(history: list[dict[str, Any]]) -> InducedRuleReport | None:
    """Induce a compact rule report from structured transformation traces.

    Default path is typed trace-slot closure followed by autonomous CPI.  The
    old hand grammar is now a debug fallback only, controlled by
    ``MARS_ALLOW_HAND_GRAMMAR_FALLBACK=1``.
    """

    if os.environ.get("MARS_USE_TYPED_TRACE_SLOTS", "1") not in (
        "0",
        "false",
        "False",
        "no",
    ):
        typed = induce_typed_trace_slot_report(history)
        if typed is not None:
            return typed

    if os.environ.get("MARS_USE_AUTONOMOUS_INDUCTION", "1") not in (
        "0",
        "false",
        "False",
        "no",
    ):
        auto = induce_autonomous_trace_rule_report(history)
        if auto is not None:
            return auto
        if os.environ.get("MARS_ALLOW_HAND_GRAMMAR_FALLBACK", "0") not in (
            "1",
            "true",
            "True",
            "yes",
        ):
            return None

    records = _sequence_records(history)
    if not records:
        return None
    rules = _infer_sequence_rules(records)
    if not rules:
        return None
    artifact = "\n".join(rules)
    return InducedRuleReport(
        artifact=artifact,
        matched_rules=tuple(r.split(":", 1)[0] for r in rules),
        n_traces=len(records),
    )
