import unittest

from mars.induction.universal_cpi import HypothesisProgram, ProgramScore
from mars.skills import (
    aggregate_supermetrics,
    compute_cpi_supermetrics,
    official_gap_metric,
)


class SuperMetricTests(unittest.TestCase):
    def test_good_cpi_result_gets_high_universal_score(self):
        program = HypothesisProgram(
            name="typed_prior_rule",
            description="typed prior recovered exact rule with evidence",
            code="def rule(x, ctx):\n    return x\n",
            fn=lambda x, ctx: x,
            complexity=1.0,
            tags=("typed_prior", "self_layer_program"),
        )
        score = ProgramScore(
            name="typed_prior_rule",
            loss_mean=0.0,
            exact_rate=1.0,
            mdl_score=0.05,
            n_scored=5,
            complexity=1.0,
        )
        report = compute_cpi_supermetrics(
            benchmark="toy",
            n_observations=5,
            n_proposed=4,
            n_valid=4,
            winners=[(program, score)],
            report="HYPOTHESIS: exact. WORKFLOW SUMMARY: Evidence: value=1 posterior=9.",
            errors=["promoted self layer typed_prior_rule for toy"],
            wall_time_s=0.1,
        )

        self.assertGreater(report.universal_score, 0.8)
        self.assertEqual(report.execution_validity, 1.0)
        self.assertEqual(report.exactness, 1.0)
        self.assertGreater(report.transfer_reuse, 0.0)

    def test_failed_cpi_result_gets_low_universal_score(self):
        report = compute_cpi_supermetrics(
            benchmark="toy",
            n_observations=5,
            n_proposed=6,
            n_valid=0,
            winners=[],
            report="",
            errors=["candidate_0: SyntaxError"],
        )

        self.assertLess(report.universal_score, 0.25)
        self.assertEqual(report.execution_validity, 0.0)
        self.assertEqual(report.heldout_fit, 0.0)

    def test_aggregate_and_official_gap_are_benchmark_agnostic(self):
        a = compute_cpi_supermetrics(
            benchmark="a",
            n_observations=3,
            n_proposed=1,
            n_valid=1,
            winners=[
                (
                    HypothesisProgram("p", "p", "def f(x): return x", lambda x: x),
                    ProgramScore("p", 0.0, 1.0, 0.1, 3, 1.0),
                )
            ],
            report="Evidence: exact value=1",
        )
        b = compute_cpi_supermetrics(
            benchmark="b",
            n_observations=3,
            n_proposed=1,
            n_valid=0,
            winners=[],
            report="",
        )
        agg = aggregate_supermetrics([a, b])
        self.assertEqual(agg["n"], 2)
        self.assertIn("universal_score", agg["by_metric"])

        gap = official_gap_metric(weak_score=10, strong_score=20, system_score=25)
        self.assertTrue(gap["beats_strong"])
        self.assertGreater(gap["progress_weak_to_strong"], 1.0)

    def test_aggregate_supermetrics_summarizes_epistemic_certificates(self):
        rows = [
            {
                "benchmark": "a",
                "universal_score": 0.9,
                "execution_validity": 1.0,
                "heldout_fit": 1.0,
                "exactness": 1.0,
                "mdl_efficiency": 0.9,
                "posterior_concentration": 1.0,
                "residual_localization": 1.0,
                "transfer_reuse": 0.5,
                "report_grounding": 1.0,
                "epistemic_certificate": {
                    "status": "accepted",
                    "universal_score": 3.0,
                    "compression_gain": 4.0,
                    "leakage_penalty": 0.0,
                    "transfer_support": 0.5,
                    "fingerprint": {"interface_family": "numeric_law"},
                },
            },
            {
                "benchmark": "b",
                "universal_score": 0.4,
                "execution_validity": 1.0,
                "heldout_fit": 0.5,
                "exactness": 0.0,
                "mdl_efficiency": 0.6,
                "posterior_concentration": 0.5,
                "residual_localization": 0.5,
                "transfer_reuse": 0.0,
                "report_grounding": 0.4,
                "epistemic_certificate": {
                    "status": "rejected",
                    "universal_score": -0.5,
                    "compression_gain": 0.1,
                    "leakage_penalty": 0.6,
                    "transfer_support": 0.0,
                    "fingerprint": {"interface_family": "table_analysis"},
                },
            },
        ]

        agg = aggregate_supermetrics(rows)

        self.assertEqual(agg["epistemic"]["n"], 2)
        self.assertEqual(agg["epistemic"]["accepted"], 1)
        self.assertEqual(agg["epistemic"]["rejected"], 1)
        self.assertEqual(agg["epistemic"]["acceptance_rate"], 0.5)
        self.assertEqual(agg["epistemic"]["mean_leakage_penalty"], 0.3)
        self.assertEqual(agg["epistemic"]["interface_families"]["numeric_law"], 1)


if __name__ == "__main__":
    unittest.main()
