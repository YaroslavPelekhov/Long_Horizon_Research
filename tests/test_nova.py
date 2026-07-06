import os
import unittest
from unittest.mock import patch

import pandas as pd

from mars.induction.universal_cpi import CPIAdapter, Observation, UniversalCPI
from mars.induction.cpi_adapters import DiscoveryAdapter
from mars.nova import NOVASynthesizer, build_oracle_curriculum


class NOVAStringSuffixAdapter(CPIAdapter):
    name = "nova_suffix_test"

    def interface_description(self) -> str:
        return "Toy string transition task."

    def signature_hint(self) -> str:
        return "def rule(current: str, context: dict) -> str"

    def collect_observations(self):
        return [
            Observation(inputs="al", target="alzz", context={}),
            Observation(inputs="be", target="bezz", context={}),
            Observation(inputs="cy", target="cyzz", context={}),
            Observation(inputs="do", target="dozz", context={}),
        ]

    def execute(self, program, obs):
        return program.fn(obs.inputs, obs.context)

    def loss(self, prediction, obs) -> float:
        return 0.0 if prediction == obs.target else 1.0

    def render_report(self, winners, observations) -> str:
        return winners[0][0].description if winners else ""


class NOVAPowerLawAdapter(CPIAdapter):
    name = "nova_power_test"

    def interface_description(self) -> str:
        return "Toy numeric power-law task."

    def signature_hint(self) -> str:
        return "def law(inputs: dict) -> float"

    def collect_observations(self):
        rows = []
        for m1, m2, r in [
            (2.0, 3.0, 2.0),
            (4.0, 5.0, 2.0),
            (6.0, 2.0, 3.0),
            (3.0, 7.0, 1.0),
            (5.0, 4.0, 4.0),
            (8.0, 3.0, 2.0),
        ]:
            target = 2.0 * m1 * m2 / (r ** 1.5)
            rows.append(
                Observation(
                    inputs={"m1": m1, "m2": m2, "r": r},
                    target=target,
                    context={},
                )
            )
        return rows

    def execute(self, program, obs):
        return program.fn(obs.inputs)

    def loss(self, prediction, obs) -> float:
        rel = abs(float(prediction) - float(obs.target)) / (abs(float(obs.target)) + 1e-9)
        return min(1.0, rel)

    def render_report(self, winners, observations) -> str:
        return winners[0][0].description if winners else ""


class NOVATests(unittest.TestCase):
    def test_oracle_curriculum_is_built_from_observations(self):
        curriculum = build_oracle_curriculum(
            NOVAStringSuffixAdapter().collect_observations(),
            interface_name="unit",
        )
        self.assertEqual(len(curriculum.tasks), 4)
        self.assertEqual(curriculum.target_kind(), "string")
        self.assertEqual(curriculum.tasks[0].target, "alzz")

    def test_synthesizer_learns_suffix_from_local_oracle_tasks(self):
        observations = NOVAStringSuffixAdapter().collect_observations()
        result = NOVASynthesizer(max_programs=8).synthesize(
            signature_hint="def rule(current: str, context: dict) -> str",
            observations=observations,
            interface_name="unit",
        )
        names = [item["name"] for item in result.program_sources]
        self.assertIn("nova_append_zz", names)
        best = result.program_sources[0]
        self.assertEqual(best["name"], "nova_append_zz")
        self.assertAlmostEqual(float(best["nova_loss"]), 0.0)

    def test_universal_cpi_uses_nova_without_llm_round(self):
        env = {
            "MARS_NOVA": "1",
            "MARS_DPSR": "0",
            "MARS_SELF_WRITE_LAYERS": "0",
            "MARS_SELF_WRITE_MODULES": "0",
            "MARS_RESIDUAL_OPERATOR_LAYERS": "0",
        }
        with patch.dict("os.environ", env, clear=False):
            result = UniversalCPI(model="unused", max_rounds=0, n_proposals=0).run(
                NOVAStringSuffixAdapter()
            )
        self.assertEqual(result.rounds, 0)
        self.assertEqual(result.winners[0][1].loss_mean, 0.0)
        self.assertIn("nova", result.winners[0][0].tags)
        self.assertTrue(any(p.get("kind") == "nova" for p in result.proposals_raw))

    def test_universal_cpi_nova_learns_fractional_power_law(self):
        env = {
            "MARS_NOVA": "1",
            "MARS_DPSR": "0",
            "MARS_SELF_WRITE_LAYERS": "0",
            "MARS_SELF_WRITE_MODULES": "0",
            "MARS_RESIDUAL_OPERATOR_LAYERS": "0",
        }
        with patch.dict("os.environ", env, clear=False):
            result = UniversalCPI(model="unused", max_rounds=0, n_proposals=0).run(
                NOVAPowerLawAdapter()
            )
        self.assertEqual(result.rounds, 0)
        self.assertLess(result.winners[0][1].loss_mean, 1e-8)
        self.assertIn("nova", result.winners[0][0].tags)
        self.assertIn("1.5", result.winners[0][0].code)

    def test_nova_table_semantic_analyzers_beat_shape_under_discovery_loss(self):
        df = pd.DataFrame(
            {
                "year": [2000, 2001, 2002, 2003, 2004, 2005],
                "education spending": [1, 2, 3, 4, 5, 6],
                "labor force": [10, 12, 14, 16, 18, 20],
                "gdp per capita": [100, 130, 160, 190, 220, 250],
                "noise": [3, 1, 4, 1, 5, 9],
            }
        )
        question = "How does education spending affect labor force and GDP per capita?"
        adapter = DiscoveryAdapter(
            df,
            question,
            "Education can affect human capital and economic output.",
            {
                "education spending": "government education expenditure",
                "labor force": "human capital proxy",
                "gdp per capita": "economic output",
            },
            enabled_modules=set(),
        )
        observations = adapter.collect_observations()
        result = NOVASynthesizer(max_programs=8).synthesize(
            signature_hint=adapter.signature_hint(),
            observations=observations,
            interface_name=adapter.name,
            question=question,
            column_descriptions=adapter.column_descriptions,
            domain_context=adapter.domain_knowledge,
        )
        names = [item["name"] for item in result.program_sources]
        self.assertIn("nova_table_pair_association", names)
        self.assertIn("nova_table_mediated_chain", names)

        engine = UniversalCPI(model="unused", max_rounds=0, n_proposals=0)
        with patch.dict(os.environ, {"MARS_NOVA": "1"}, clear=False):
            programs, errors, _props = engine._nova_programs(adapter, observations)
        self.assertFalse(errors)
        ranked = engine._rank(programs, observations, adapter)
        self.assertNotEqual(ranked[0][0].name, "nova_table_shape")
        self.assertLess(ranked[0][1].loss_mean, 0.5)


if __name__ == "__main__":
    unittest.main()
