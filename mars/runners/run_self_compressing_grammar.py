"""Run a benchmark-agnostic demo of self-compressing hypothesis skills.

This runner intentionally avoids benchmark names and benchmark-specific
adapters.  It shows the general mechanism:

  residuals -> local failure-born skills -> anti-unified parent grammar
"""

from __future__ import annotations

import json

from mars.skills import ResidualCase, SkillGrammar, expr_to_text


def _threshold_batch(feature: str, cutoff: float, low_fails: bool = True) -> list[ResidualCase]:
    cases: list[ResidualCase] = []
    for value in (2, 5, 9, 18, 24, 31):
        failed = value <= cutoff if low_fails else value > cutoff
        cases.append(
            ResidualCase(
                features={feature: value, "noise": value % 3},
                prediction="old_rule",
                target="new_rule" if failed else "old_rule",
                loss=1.0 if failed else 0.0,
            )
        )
    return cases


def _interaction_batch(left: str, right: str) -> list[ResidualCase]:
    cases: list[ResidualCase] = []
    for a, b in ((1, 2), (2, 3), (3, 5), (8, 13)):
        cases.append(
            ResidualCase(
                features={left: a, right: b, "bias": 1},
                prediction=a,
                target=a * b,
                loss=1.0,
            )
        )
    return cases


def main() -> None:
    grammar = SkillGrammar(min_gain=0.25)
    report = grammar.induce(
        [
            ("energy_failure_skill", _threshold_batch("energy", 12)),
            ("age_failure_skill", _threshold_batch("age", 12)),
            ("temperature_failure_skill", _threshold_batch("temperature", 12)),
            ("mass_pair_skill", _interaction_batch("mass_a", "mass_b")),
            ("charge_pair_skill", _interaction_batch("charge_a", "charge_b")),
        ]
    )
    payload = {
        "born": [
            {
                "name": s.name,
                "schema": expr_to_text(s.schema),
                "detector": s.detector,
                "generator": s.generator,
                "support": s.support,
            }
            for s in report.born
        ],
        "consolidated": [
            {
                "name": s.name,
                "schema": expr_to_text(s.schema),
                "compression_cost": round(s.compression_cost, 3),
                "children": list(s.parents),
                "contract": {
                    "detector": s.detector,
                    "generator": s.generator,
                    "verifier": s.verifier,
                    "repair": s.repair,
                },
            }
            for s in report.consolidated
        ],
        "rejected": list(report.rejected),
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
