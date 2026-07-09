import unittest


class AnswerPlanInducerTests(unittest.TestCase):
    def test_paired_measure_plan_filters_requested_domain(self):
        import pandas as pd

        from mars.skills import infer_answer_plan_hypothesis

        df = pd.DataFrame(
            {
                "project.x": ["Experimental Economics", "Experimental Economics", "Psychology", "Psychology"],
                "fiso": [0.60, 0.54, 0.52, 0.48],
                "fisr": [0.34, 0.28, 0.25, 0.23],
            }
        )
        result = infer_answer_plan_hypothesis(
            question=(
                "In Experimental Economics, what is the average effect estimate on the "
                "Fisher-z scale in original studies compared with replication studies?"
            ),
            domain_context="",
            df=df,
            column_descriptions={
                "project.x": "scientific domain or project",
                "fiso": "original study effect estimate on Fisher-z scale",
                "fisr": "replication study effect estimate on Fisher-z scale",
            },
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("answer_plan_paired_measure", result.evidence)
        self.assertIn("Experimental Economics", result.hypothesis)
        self.assertIn("0.57", result.hypothesis)
        self.assertIn("0.31", result.hypothesis)
        self.assertNotIn("Psychology", result.hypothesis)

    def test_paired_measure_plan_renders_which_domains(self):
        import pandas as pd

        from mars.skills import infer_answer_plan_hypothesis

        df = pd.DataFrame(
            {
                "project.x": ["Experimental Economics", "Experimental Economics", "Psychology", "Psychology", "Other", "Other"],
                "fiso": [0.60, 0.54, 0.52, 0.48, 0.10, 0.14],
                "fisr": [0.34, 0.28, 0.25, 0.23, 0.20, 0.18],
            }
        )
        result = infer_answer_plan_hypothesis(
            question=(
                "Which domains have larger effect size estimates in original studies "
                "than in replication studies?"
            ),
            domain_context="",
            df=df,
            column_descriptions={
                "project.x": "scientific domain or project",
                "fiso": "original study effect estimate on Fisher-z scale",
                "fisr": "replication study effect estimate on Fisher-z scale",
            },
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("Experimental Economics", result.hypothesis)
        self.assertIn("Psychology", result.hypothesis)
        self.assertNotIn("Other", result.hypothesis)
        self.assertIn("answer_form=grouped_original_replication_comparison", result.workflow)

    def test_paired_measure_plan_can_render_local_plus_global_scope(self):
        import pandas as pd

        from mars.skills import infer_answer_plan_hypothesis

        df = pd.DataFrame(
            {
                "project.x": ["Experimental Economics", "Experimental Economics", "Psychology", "Psychology"],
                "fiso": [0.60, 0.54, 0.52, 0.48],
                "fisr": [0.34, 0.28, 0.25, 0.23],
            }
        )
        result = infer_answer_plan_hypothesis(
            question=(
                "In Experimental Economics, what is the average effect estimate in original studies "
                "as compared to that in replication studies?"
            ),
            domain_context="",
            df=df,
            column_descriptions={
                "project.x": "scientific domain or project",
                "fiso": "original study effect estimate",
                "fisr": "replication study effect estimate",
            },
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("Experimental Economics", result.hypothesis)
        self.assertIn("Psychology", result.hypothesis)
        self.assertIn("scope=scope_local_plus_global", result.evidence)

    def test_grouped_prevalence_plan_closes_habitat_profile(self):
        import pandas as pd

        from mars.skills import infer_answer_plan_hypothesis

        df = pd.DataFrame(
            {
                "habitat": ["forest", "forest", "urban", "urban", "cropland"],
                "n.gard": [4, 6, 2, 1, 0],
                "n.total": [5, 5, 10, 10, 4],
            }
        )
        result = infer_answer_plan_hypothesis(
            question="How does the prevalence of plants introduced via gardening vary based on habitat type?",
            domain_context="",
            df=df,
            column_descriptions={
                "habitat": "habitat type",
                "n.gard": "count of species introduced through Gardening pathway",
                "n.total": "total count of introduced species",
            },
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("answer_plan_grouped_prevalence", result.evidence)
        self.assertIn("urban/cropland habitats", result.hypothesis)
        self.assertIn("natural habitats", result.hypothesis)
        self.assertIn("abstraction=human_modified_vs_natural", result.evidence)
        self.assertIn("answer_form=grouped_prevalence_profile", result.workflow)


if __name__ == "__main__":
    unittest.main()
