import unittest

from mars.skills import compile_benchmark_theory
from mars.skills.theory_runtime import render_theory_context


class BenchmarkTheoryTests(unittest.TestCase):
    def test_discovery_like_sources_birth_metric_and_schema_skills(self):
        theory = compile_benchmark_theory(
            "unseen_data_discovery",
            {
                "readme": (
                    "Each task contains a goal, dataset csv files, metadata, "
                    "schema descriptions, and a natural-language hypothesis. "
                    "The evaluator reports HMS using context, variables, and relation match."
                )
            },
        )
        self.assertIn("datasets", theory.observable_inputs)
        self.assertIn("hypothesis", theory.target_outputs)
        self.assertIn("HMS", [m.name for m in theory.metrics])
        self.assertIn("metric_slot_mismatch", [r.name for r in theory.residuals])
        self.assertIn("schema_grounding_skill", [s.name for s in theory.skill_schemas])

    def test_rollout_sources_birth_transition_simulator_skill(self):
        theory = compile_benchmark_theory(
            "sequence_world",
            {
                "paper": (
                    "The agent interacts with a long-horizon environment. "
                    "It observes state trajectories and sequences, induces a transition rule, "
                    "rolls out candidates, and receives a final score."
                )
            },
        )
        self.assertIn("rollout_candidate", [a.name for a in theory.actions])
        self.assertIn("rollout_drift", [r.name for r in theory.residuals])
        self.assertIn("transition_simulator_skill", [s.name for s in theory.skill_schemas])

    def test_theory_does_not_depend_on_benchmark_name_only(self):
        source = {
            "generic": (
                "A task has python files and tools. The program must execute, "
                "save the required output, and is measured by valid execution VER and success rate SR."
            )
        }
        a = compile_benchmark_theory("name_a", source)
        b = compile_benchmark_theory("name_b", source)
        self.assertEqual([m.name for m in a.metrics], [m.name for m in b.metrics])
        self.assertEqual([r.name for r in a.residuals], [r.name for r in b.residuals])
        self.assertIn("execution_preflight_skill", [s.name for s in a.skill_schemas])

    def test_theory_runtime_context_names_residual_and_skill(self):
        theory = compile_benchmark_theory(
            "sequence_world",
            {
                "paper": (
                    "The agent interacts with a long-horizon environment, observes "
                    "state trajectories, induces a transition rule, rolls out candidates, "
                    "and receives a final_score."
                )
            },
        )
        context = render_theory_context(theory)
        self.assertIn("SELF-INDUCED BENCHMARK THEORY", context)
        self.assertIn("rollout_drift", context)
        self.assertIn("transition_simulator_skill", context)


if __name__ == "__main__":
    unittest.main()
