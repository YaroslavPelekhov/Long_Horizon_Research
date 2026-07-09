import unittest


class ContrastiveWorldInducerTests(unittest.TestCase):
    def test_contrastive_growth_world_beats_peak_world(self):
        import pandas as pd

        from mars.skills import infer_contrastive_world_hypothesis

        years = list(range(-2000, -1099))
        growth = [1.0 if -1505 <= y <= -1325 else 0.1 for y in years]
        peak_value = list(range(len(years)))
        df = pd.DataFrame({"CE": years, "g_all_mean": growth, "kde_all_mean": peak_value})

        result = infer_contrastive_world_hypothesis(
            question="In what centuries did we see the highest growth phase of the period between 2000 BCE and 1100 BCE?",
            domain_context="",
            data=df,
            column_descriptions={"CE": "calendar year", "g_all_mean": "mean growth rate"},
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("1500 BCE", result.hypothesis)
        self.assertIn("growth", result.hypothesis)
        self.assertIn("world=", result.workflow)

    def test_stated_nested_delta_collapses_variable_identity(self):
        from mars.skills import infer_contrastive_world_hypothesis

        result = infer_contrastive_world_hypothesis(
            question=(
                "The effect of which variable on BA degree completion decreases from "
                "0.3636 to -0.2293 when both race and academic characteristics are included?"
            ),
            domain_context="",
            data=None,
            column_descriptions={},
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("SES", result.hypothesis)
        self.assertIn("0.3636", result.hypothesis)
        self.assertIn("-0.2293", result.hypothesis)


if __name__ == "__main__":
    unittest.main()
