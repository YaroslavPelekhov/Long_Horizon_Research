import unittest


class ProblemFrameInducerTests(unittest.TestCase):
    def test_growth_window_uses_interval_not_peak(self):
        import pandas as pd

        from mars.skills import infer_problem_frame_hypothesis

        years = list(range(-2000, -1099))
        growth = []
        for y in years:
            # A broad high-growth phase around 1500-1300 BCE.
            growth.append(1.0 if -1505 <= y <= -1325 else 0.1)
        df = pd.DataFrame({"CE": years, "g_all_mean": growth, "kde_all_mean": range(len(years))})

        result = infer_problem_frame_hypothesis(
            question="In what centuries did we see the highest growth phase of the period between 2000 BCE and 1100 BCE?",
            domain_context="",
            data=df,
            column_descriptions={"CE": "calendar year CE/BCE", "g_all_mean": "mean growth rate"},
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("growth phase", result.hypothesis)
        self.assertIn("problem_frame_growth_window", result.evidence)
        self.assertNotIn("peaked", result.hypothesis.lower())

    def test_nested_model_delta_measures_coefficient_shift(self):
        import numpy as np
        import pandas as pd

        from mars.skills import infer_problem_frame_hypothesis

        rng = np.random.default_rng(7)
        n = 260
        ses = rng.normal(size=n)
        ability = ses * 1.8 + rng.normal(scale=0.35, size=n)
        percentile = ability * 20 + rng.normal(scale=5, size=n)
        race = np.where(rng.normal(size=n) > 0, "White", "Black")
        logits = -1.3 + 0.4 * ses + 1.2 * ability + (race == "White") * 0.2
        p = 1 / (1 + np.exp(-logits))
        y = rng.random(n) < p
        df = pd.DataFrame(
            {
                "SAMPLE_RACE": race,
                "ABILITY: COMPOSITE OF ASVAB SCORE": ability,
                "PERCENTILE IN CLASS": percentile,
                "BA DEGREE COMPLETED": y,
                "SES": ses,
            }
        )

        result = infer_problem_frame_hypothesis(
            question="How does the effect of SES on BA Degree completion change when both race and academic characteristics are considered as compared to when only race is considered?",
            domain_context="",
            data=df,
            column_descriptions={
                "SAMPLE_RACE": "Race of the respondent",
                "ABILITY: COMPOSITE OF ASVAB SCORE": "Academic ability score",
                "PERCENTILE IN CLASS": "Academic class percentile",
                "BA DEGREE COMPLETED": "Boolean BA degree completed",
                "SES": "Socioeconomic status",
            },
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("SES", result.hypothesis)
        self.assertIn("problem_frame_nested_model_delta", result.evidence)

    def test_wide_panel_influence_claim_respects_measured_sign(self):
        import pandas as pd

        from mars.skills import infer_problem_frame_hypothesis

        df = pd.DataFrame(
            {
                "Country Group": ["Lower middle income", "Lower middle income"],
                "Series Name": [
                    "Adjusted savings: education expenditure (percentage of GNI)",
                    "GNI per capita (constant 2015 USdollar)",
                ],
                "1990 [YR1990]": [5.0, 100.0],
                "1991 [YR1991]": [6.0, 90.0],
                "1992 [YR1992]": [7.0, 80.0],
                "1993 [YR1993]": [8.0, 70.0],
            }
        )

        result = infer_problem_frame_hypothesis(
            question="How does increased education expenditure influence per capita GDP?",
            domain_context="",
            data=df,
            column_descriptions={},
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("does not support a positive", result.hypothesis)
        self.assertNotIn("positive impact", result.hypothesis)
        self.assertIn("r=-1", result.evidence)


if __name__ == "__main__":
    unittest.main()
