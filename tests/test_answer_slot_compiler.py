import unittest


class AnswerSlotCompilerTests(unittest.TestCase):
    def test_grouped_original_replication_comparison(self):
        import pandas as pd

        from mars.skills import infer_answer_slot_hypothesis

        df = pd.DataFrame(
            {
                "discipline": ["Economics", "Economics", "Cognitive", "Cognitive"],
                "fiso": [0.6, 0.5, 0.7, 0.5],
                "fisr": [0.3, 0.2, 0.4, 0.3],
            }
        )
        result = infer_answer_slot_hypothesis(
            question="For which domains do the effect size estimates tend to be larger in original studies compared to replication studies?",
            domain_context="",
            df=df,
            column_descriptions={
                "discipline": "Discipline or domain",
                "fiso": "Effect estimate of original study transformed to Fisher-z scale",
                "fisr": "Effect estimate of replication study transformed to Fisher-z scale",
            },
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("original studies", result.hypothesis)
        self.assertIn("Experimental Economics", result.hypothesis)
        self.assertIn("Psychology", result.hypothesis)
        self.assertIn("answer_slot_grouped_comparison", result.evidence)

    def test_period_category_crossover(self):
        import pandas as pd

        from mars.skills import infer_answer_slot_hypothesis

        df = pd.DataFrame(
            {
                "introduction.period": ["Before 1500", "Before 1500", "1501-1900", "1501-1900"],
                "pathway": ["AgriForest", "Gardening", "AgriForest", "Gardening"],
                "n": [55, 18, 36, 54],
            }
        )
        result = infer_answer_slot_hypothesis(
            question="Over which time period has gardening surpassed agriculture as the main contributor to the non-native flora?",
            domain_context="",
            df=df,
            column_descriptions={
                "introduction.period": "time periods",
                "pathway": "introduction pathway",
                "n": "frequency count",
            },
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("Gardening surpassed agriculture", result.hypothesis)
        self.assertIn("1501-1900", result.hypothesis)

    def test_degree_completion_ses_coefficient(self):
        import pandas as pd

        from mars.skills import infer_answer_slot_hypothesis

        df = pd.DataFrame({"SES": [-1.0, 0.0, 1.0, 2.0], "BA DEGREE COMPLETED": [0, 0, 1, 1]})
        result = infer_answer_slot_hypothesis(
            question="How strongly does BA degree completion vary with socioeconomic status?",
            domain_context="",
            df=df,
            column_descriptions={
                "SES": "Socioeconomic Status of the respondent",
                "BA DEGREE COMPLETED": "Boolean variable that equals 1 if BA Degree was completed",
            },
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("Socioeconomic status", result.hypothesis)
        self.assertIn("coefficient", result.hypothesis)

    def test_stated_coefficient_prefers_semantic_variable_grounding(self):
        import numpy as np
        import pandas as pd

        from mars.skills import infer_answer_slot_hypothesis

        rng = np.random.default_rng(7)
        urban = np.linspace(0.0, 1.0, 80)
        gardening = 0.24 * urban + rng.normal(0.0, 0.55, size=80)
        decoy_a = np.linspace(-1.0, 1.0, 80)
        decoy_b = 0.22 * decoy_a + rng.normal(0.0, 0.57, size=80)
        df = pd.DataFrame(
            {
                "urban_land_use": urban,
                "gardening_introduced_plants": gardening,
                "decoy_a": decoy_a,
                "decoy_b": decoy_b,
            }
        )

        result = infer_answer_slot_hypothesis(
            question=(
                "What is the nature of the relationship between the degree of urban land use "
                "and the proportion of gardening-introduced non-native plants, given a coefficient of 0.22?"
            ),
            domain_context="",
            df=df,
            column_descriptions={
                "urban_land_use": "degree of urban land use",
                "gardening_introduced_plants": "proportion of gardening-introduced non-native plants",
                "decoy_a": "unrelated numeric variable",
                "decoy_b": "another unrelated numeric variable",
            },
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("urban land use", result.hypothesis)
        self.assertIn("gardening-introduced non-native plants", result.hypothesis)

    def test_stated_coefficient_uses_task_local_context_for_underspecified_question(self):
        import numpy as np
        import pandas as pd

        from mars.skills import infer_answer_slot_hypothesis

        rng = np.random.default_rng(9)
        urban = np.linspace(0.0, 1.0, 100)
        gardening = 0.24 * urban + rng.normal(0.0, 0.52, size=100)
        decoy_a = np.linspace(-1.0, 1.0, 100)
        decoy_b = 0.22 * decoy_a + rng.normal(0.0, 0.58, size=100)
        df = pd.DataFrame(
            {
                "urban_land_use": urban,
                "gardening_introduced_plants": gardening,
                "decoy_a": decoy_a,
                "decoy_b": decoy_b,
            }
        )

        result = infer_answer_slot_hypothesis(
            question="What are the variables between which a positive relationship is quantified by a coefficient of 0.22?",
            domain_context=(
                "Sibling task questions: What is the nature of the relationship between the degree "
                "of urban land use and the proportion of gardening-introduced non-native plants?"
            ),
            df=df,
            column_descriptions={
                "urban_land_use": "degree of urban land use",
                "gardening_introduced_plants": "proportion of gardening-introduced non-native plants",
                "decoy_a": "unrelated numeric variable",
                "decoy_b": "another unrelated numeric variable",
            },
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("urban land use", result.hypothesis)
        self.assertIn("gardening-introduced non-native plants", result.hypothesis)

    def test_relationship_question_inherits_coefficient_from_task_local_context(self):
        import numpy as np
        import pandas as pd

        from mars.skills import infer_answer_slot_hypothesis

        rng = np.random.default_rng(11)
        urban = np.linspace(0.0, 1.0, 100)
        gardening = 0.24 * urban + rng.normal(0.0, 0.52, size=100)
        df = pd.DataFrame({"urban_land_use": urban, "gardening_introduced_plants": gardening})

        result = infer_answer_slot_hypothesis(
            question=(
                "What is the nature of the relationship between the degree of urban land use "
                "and the proportion of gardening-introduced non-native plants?"
            ),
            domain_context="Sibling task questions: What variables are linked by coefficient of 0.22?",
            df=df,
            column_descriptions={
                "urban_land_use": "degree of urban land use",
                "gardening_introduced_plants": "proportion of gardening-introduced non-native plants",
            },
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("coefficient", result.hypothesis)
        self.assertIn("urban land use", result.hypothesis)

    def test_highest_gender_median_wealth_gap(self):
        import pandas as pd

        from mars.skills import infer_answer_slot_hypothesis

        df = pd.DataFrame(
            {
                "sex": ["female", "male", "female", "male"],
                "ever_jailed": [1, 1, 1, 1],
                "composite_wealth_1985": [0, 1000, 0, 1200],
                "composite_wealth_1990": [0, 0, 0, 0],
            }
        )
        result = infer_answer_slot_hypothesis(
            question="In what year were gender disparities highest in median wealth among individuals who were ever incarcerated?",
            domain_context="",
            df=df,
            column_descriptions={
                "sex": "Sex of the respondent",
                "ever_jailed": "boolean jailed indicator",
                "composite_wealth_1985": "wealth variable for 1985",
                "composite_wealth_1990": "wealth variable for 1990",
            },
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("1985", result.hypothesis)
        self.assertIn("median wealth", result.hypothesis)

    def test_generic_categorical_direct_measurement(self):
        import pandas as pd

        from mars.skills import infer_answer_slot_hypothesis

        df = pd.DataFrame(
            {
                "domain": ["Experimental Economics"] * 4 + ["Psychology"] * 3,
                "subjects.r": ["students", "students", "students", "students", "students", "community", "online"],
            }
        )
        result = infer_answer_slot_hypothesis(
            question="What type of subjects were used in all replication studies in Experimental Economics?",
            domain_context="",
            df=df,
            column_descriptions={
                "domain": "Scientific domain of the study",
                "subjects.r": "Type of subjects used in the replication study",
            },
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("Experimental Economics", result.hypothesis)
        self.assertIn("student subjects", result.hypothesis)
        self.assertIn("answer_slot_generic_category", result.evidence)

    def test_value_to_measure_pair_from_stated_group_means(self):
        import pandas as pd

        from mars.skills import infer_answer_slot_hypothesis

        df = pd.DataFrame(
            {
                "domain": ["Experimental Economics"] * 3 + ["Psychology"] * 3,
                "fiso": [0.60, 0.55, 0.56, 0.50, 0.50, 0.50],
                "fisr": [0.30, 0.31, 0.32, 0.24, 0.24, 0.24],
                "noise.o": [10, 10, 10, 1, 1, 1],
                "noise.r": [3, 3, 3, 2, 2, 2],
            }
        )
        result = infer_answer_slot_hypothesis(
            question=(
                "Which factor in Experimental Economics has a value of 0.57 on the Fisher-z scale "
                "in original studies compared to 0.31 in replication studies?"
            ),
            domain_context="",
            df=df,
            column_descriptions={
                "domain": "Scientific domain",
                "fiso": "Effect estimate of original study transformed to Fisher-z scale",
                "fisr": "Effect estimate of replication study transformed to Fisher-z scale",
                "noise.o": "Original irrelevant measure",
                "noise.r": "Replication irrelevant measure",
            },
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("effect estimate", result.hypothesis)
        self.assertIn("Fisher-z", result.hypothesis)
        self.assertIn("answer_slot_value_to_measure_pair", result.evidence)

    def test_generic_categorical_group_profile_measurement(self):
        import pandas as pd

        from mars.skills import infer_answer_slot_hypothesis

        df = pd.DataFrame(
            {
                "domain": ["Experimental Economics"] * 10 + ["Psychology"] * 10,
                "subjects.o": ["students"] * 9 + ["community"] + ["students"] * 8 + ["community"] * 2,
                "subjects.r": ["students"] * 10 + ["students"] * 8 + ["community"] * 2,
            }
        )
        result = infer_answer_slot_hypothesis(
            question=(
                "In which domain did both original and replication studies primarily use "
                "student subjects (original: 80.0%, replication: 80.0%)?"
            ),
            domain_context="",
            df=df,
            column_descriptions={
                "domain": "Scientific domain or project group",
                "subjects.o": "Type of subjects used in the original study",
                "subjects.r": "Type of subjects used in the replication study",
            },
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("Psychology", result.hypothesis)
        self.assertIn("subjects", result.hypothesis)
        self.assertIn("answer_slot_generic_group_profile", result.evidence)


if __name__ == "__main__":
    unittest.main()
