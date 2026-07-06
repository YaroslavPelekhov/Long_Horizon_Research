import unittest

from mars.skills import (
    CompiledHypothesis,
    HypothesisWorkbenchReport,
    InterfaceObservation,
    MetricContract,
    compile_hypothesis_workbench,
    compile_interface_contract,
    discoverybench_contracts,
    infer_temporal_event_hypothesis,
    residuals_from_score,
    score_contract,
    score_interface_contract,
)


class MetricCompilerTests(unittest.TestCase):
    def test_hypothesis_workbench_generates_and_repairs_candidates(self):
        evidence = [
            "[rk_sketch_causal_evidence_chain] wide-mode cause=Adjusted savings: education expenditure; "
            "wide corr(cause,mediator Labor force participation rate)=0.75; "
            "wide corr(cause,outcome GNI per capita)=-0.65; wide trend(outcome GNI per capita)=0.93",
            "[typed_table_corr] corr(1975 [YR1975],1976 [YR1976])=0.99",
        ]
        report = compile_hypothesis_workbench(
            question="How does increased education expenditure influence human capital and economic output?",
            domain_context=(
                "Labor force is a proxy for human capital. GNI per capita represents "
                "economic output. Lower middle income countries are developing countries."
            ),
            evidence_lines=evidence,
        )
        self.assertIsInstance(report, HypothesisWorkbenchReport)
        self.assertGreaterEqual(len(report.candidates), 4)
        best = report.best
        self.assertIsNotNone(best)
        assert best is not None
        text = f"{best.hypothesis} {best.workflow}".lower()
        self.assertIn("education", text)
        self.assertTrue("human" in text or "labor" in text)
        self.assertTrue("gni" in text or "economic output" in text)
        self.assertIn("hypothesis", best.compiled.output)
        self.assertGreater(best.score.weighted_score, 0.45)

    def test_temporal_event_hypothesis_detects_peak_onset_and_stability(self):
        import pandas as pd

        df = pd.DataFrame(
            {
                "CE": [-2400, -2300, -2200, -2100, -2000],
                "AxesCelts": [-1.0, 0.2, 2.0, 0.5, -0.2],
                "Dagger_inter": [-0.5, 0.1, 0.4, 0.8, 1.2],
                "ZCU_AU": [0.4, 0.3, 0.2, 0.5, 0.4],
                "Zamber": [-0.7, -0.72, -0.71, -0.69, -0.7],
                "ZMonument": [1.2, 0.2, -0.4, -0.4, -0.4],
            }
        )
        desc = {
            "AxesCelts": "Z values for Axes and Celts",
            "Dagger_inter": "Interpolated z value for Daggers",
            "ZCU_AU": "Z values for Copper and Gold",
            "Zamber": "Z values for Amber",
            "ZMonument": "Z values for Monument Count",
        }
        peak = infer_temporal_event_hypothesis(
            question="In which century did the axes become quantitatively most frequent?",
            domain_context="Economic Capital consists of Axes & Celts.",
            df=df,
            column_descriptions=desc,
        )
        self.assertIsNotNone(peak)
        assert peak is not None
        self.assertIn("axes", peak[0].lower())
        self.assertIn("2200", peak[1])

        onset = infer_temporal_event_hypothesis(
            question="In which century did the number of daggers began to increase in importance for the first time?",
            domain_context="Symbolic capital consists of Daggers.",
            df=df,
            column_descriptions=desc,
        )
        self.assertIsNotNone(onset)
        assert onset is not None
        self.assertIn("daggers", onset[0].lower())
        self.assertIn("2300", onset[1])

        stable = infer_temporal_event_hypothesis(
            question="Which social capital value stayed low and showed low fluctuation in the younger bronze age (2300-2000 BCE)?",
            domain_context="Social Capital consists of Copper and Gold, Amber, Monument Count.",
            df=df.rename(columns={"CE": "BCE"}).assign(BCE=[2400, 2300, 2200, 2100, 2000]),
            column_descriptions=desc,
        )
        self.assertIsNotNone(stable)
        assert stable is not None
        self.assertIn("monument", stable[0].lower())

    def test_relation_failure_becomes_relation_skill_residual(self):
        contract = MetricContract(
            name="db_relation_case",
            expected=CompiledHypothesis(
                text="The variable decreases in the target period.",
                context="in 1100-500 BCE",
                variables=("Belau_PC1",),
                relation="negative relationship over time",
            ),
        )
        observed = CompiledHypothesis(
            text="The variable increases in the target period.",
            context="in 1100-500 BCE",
            variables=("Belau_PC1",),
            relation="positive increase over time",
        )
        score = score_contract(contract, observed)
        self.assertIn("relation", score.failed_slots)
        residuals = residuals_from_score(score)
        relation_residual = [r for r in residuals if r.slot == "relation"][0]
        self.assertEqual(relation_residual.suggested_skill_schema[0], "relation_operator_skill")

    def test_variable_failure_becomes_entity_linker_residual(self):
        contract = MetricContract(
            name="db_variable_case",
            expected=CompiledHypothesis(
                text="subjects.o differs from subjects.r",
                context="Experimental Economics",
                variables=("subjects.o", "subjects.r"),
                relation="comparison of original and replication subjects",
            ),
        )
        observed = CompiledHypothesis(
            text="same_subjects is high",
            context="Experimental Economics",
            variables=("same_subjects",),
            relation="high consistency",
        )
        score = score_contract(contract, observed)
        self.assertIn("variables", score.failed_slots)
        residuals = residuals_from_score(score)
        self.assertTrue(any(r.suggested_skill_schema[0] == "entity_linker_skill" for r in residuals))

    def test_discoverybench_contract_builder(self):
        eval_result = {
            "gold_sub_hypo": {
                "sub_hypo": [
                    {
                        "text": "Axes peak in the fourth millennium BCE.",
                        "context": "fourth millennium BCE",
                        "variables": ["AxesCelts"],
                        "relations": "peak frequency",
                    }
                ]
            }
        }
        contracts = discoverybench_contracts(eval_result)
        self.assertEqual(len(contracts), 1)
        self.assertEqual(contracts[0].expected.variables, ("AxesCelts",))

    def test_interface_contract_scores_string_transform_without_benchmark_name(self):
        observations = [
            InterfaceObservation(inputs="ab", target="abx", context={"step": 1}),
            InterfaceObservation(inputs="cd", target="cdx", context={"step": 2}),
        ]
        contract = compile_interface_contract(observations, name="toy_string")
        self.assertEqual(contract.target_type, "string")
        self.assertTrue(any(c.kind == "exact" for c in contract.checks))

        good = score_interface_contract(contract, observations, lambda x, ctx: x + "x")
        bad = score_interface_contract(contract, observations, lambda x, ctx: x)
        self.assertEqual(good.loss_mean, 0.0)
        self.assertGreater(bad.loss_mean, 0.0)
        self.assertTrue(any(r.residual_type == "categorical_mismatch" for r in bad.residuals))

    def test_interface_contract_scores_numeric_law_without_benchmark_name(self):
        observations = [
            InterfaceObservation(inputs={"x": 2.0}, target=4.0),
            InterfaceObservation(inputs={"x": 3.0}, target=9.0),
        ]
        contract = compile_interface_contract(observations, name="toy_numeric")
        self.assertEqual(contract.target_type, "number")
        self.assertTrue(any(c.kind == "numeric" for c in contract.checks))

        good = score_interface_contract(contract, observations, lambda x, ctx: x["x"] ** 2)
        bad = score_interface_contract(contract, observations, lambda x, ctx: x["x"])
        self.assertEqual(good.loss_mean, 0.0)
        self.assertGreater(bad.loss_mean, 0.0)
        self.assertTrue(any(r.residual_type == "numeric_residual" for r in bad.residuals))

    def test_interface_contract_scores_mapping_schema_without_benchmark_name(self):
        observations = [
            InterfaceObservation(
                inputs={"a": 1},
                target={"answer": "yes", "confidence": 0.9},
            )
        ]
        contract = compile_interface_contract(observations, name="toy_mapping")
        self.assertEqual(contract.target_type, "mapping")

        score = score_interface_contract(contract, observations, lambda x, ctx: {"answer": "yes"})
        self.assertGreater(score.loss_mean, 0.0)
        self.assertTrue(any(r.residual_type == "schema_mismatch" for r in score.residuals))


if __name__ == "__main__":
    unittest.main()
