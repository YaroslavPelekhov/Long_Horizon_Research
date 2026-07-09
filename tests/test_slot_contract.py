import unittest


class SlotContractTests(unittest.TestCase):
    def test_peak_closes_variable_and_time_slots(self):
        import pandas as pd

        from mars.skills import infer_slot_contract_hypothesis

        df = pd.DataFrame(
            {
                "CE": [-2400, -2300, -2200, -2100, -2000],
                "Belau_PC1": [2.0, 3.0, 5.0, 8.0, 13.0],
                "ZBeil": [-1.0, 0.2, 2.0, 0.5, -0.2],
                "ZDolch": [-0.5, 0.1, 0.4, 0.8, 1.2],
            }
        )
        result = infer_slot_contract_hypothesis(
            question="In which century did the axes become quantitatively most frequent?",
            domain_context="Economic Capital consists of Axes and Celts.",
            df=df,
            column_descriptions={
                "Belau_PC1": "Principal component unrelated to axes",
                "ZBeil": "Z values for axes and celts",
                "ZDolch": "Z values for daggers",
            },
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.slots["variable"], "ZBeil")
        self.assertIn("axes", result.hypothesis.lower())
        self.assertIn("2200", result.workflow)

    def test_peak_uses_canonical_variable_and_coarse_bce_period(self):
        import pandas as pd

        from mars.skills import infer_slot_contract_hypothesis

        df = pd.DataFrame(
            {
                "CE": [-3300, -3200, -3100],
                "AxesCelts": [1.0, 2.0, 1.5],
                "AxesCelts_inter": [1.1, 2.6, 1.4],
            }
        )
        result = infer_slot_contract_hypothesis(
            question="In which century did the axes become quantitatively most frequent?",
            domain_context="Axes and celts are the relevant artifact class.",
            df=df,
            column_descriptions={
                "AxesCelts": "Z values for Axes and Celts",
                "AxesCelts_inter": "Interpolated z values for Axes and Celts",
            },
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.slots["variable"], "AxesCelts")
        self.assertIn("end of the 4th millennium bce", result.hypothesis.lower())
        self.assertNotIn("time=CE", result.workflow)

    def test_peak_question_uses_peak_wording(self):
        import pandas as pd

        from mars.skills import infer_slot_contract_hypothesis

        df = pd.DataFrame({"CE": [-1200, -1100, -1000], "Depot": [1.0, 3.0, 2.0], "Depot_inter": [1.1, 3.4, 2.1]})
        result = infer_slot_contract_hypothesis(
            question="In which century did the Depots peak?",
            domain_context="Economic capital contains depots.",
            df=df,
            column_descriptions={"Depot": "Depot", "Depot_inter": "Interpolated depot"},
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("around 1100 bce", result.hypothesis.lower())
        self.assertIn("peaked", result.hypothesis.lower())

    def test_simultaneous_inverse_transition(self):
        import pandas as pd

        from mars.skills import infer_slot_contract_hypothesis

        df = pd.DataFrame(
            {
                "CE": [-1900, -1800, -1700, -1600],
                "PotteryForm": [0.6, 0.7, -1.1, -1.1],
                "PotteryDecoration": [-0.6, -0.5, 1.0, 1.0],
            }
        )
        result = infer_slot_contract_hypothesis(
            question="In which century did the Diversity in Pottery Form collapses and Diversity in Pottery Decoration increases simultaneuosly?",
            domain_context="Cultural Capital consists of Diversity of Pottery form and Diversity of Pottery Decoration.",
            df=df,
            column_descriptions={
                "PotteryForm": "Z values for Pottery Form",
                "PotteryDecoration": "Z values for Pottery Decoration",
            },
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.slots["event"], "simultaneous_inverse")
        self.assertIn("around 1800 bce", result.hypothesis.lower())
        self.assertIn("collapses while", result.hypothesis.lower())

    def test_onset_uses_alias_grounding_for_daggers(self):
        import pandas as pd

        from mars.skills import infer_slot_contract_hypothesis

        df = pd.DataFrame(
            {
                "CE": [-2400, -2300, -2200, -2100, -2000],
                "ZBeil": [2.0, 3.0, 5.0, 8.0, 13.0],
                "ZDolch": [-0.5, 0.1, 0.4, 0.8, 1.2],
            }
        )
        result = infer_slot_contract_hypothesis(
            question="In which century did daggers begin to increase in importance for the first time?",
            domain_context="Symbolic capital consists of daggers.",
            df=df,
            column_descriptions={"ZBeil": "Z values for axes", "ZDolch": "Z values for daggers"},
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.slots["variable"], "ZDolch")
        self.assertIn("daggers", result.hypothesis.lower())
        self.assertIn("2300", result.workflow)

    def test_low_stability_prefers_question_relevant_social_capital(self):
        import pandas as pd

        from mars.skills import infer_slot_contract_hypothesis

        df = pd.DataFrame(
            {
                "BCE": [2400, 2300, 2200, 2100, 2000],
                "ZBeil": [2.0, 2.3, 3.0, 4.0, 5.0],
                "ZCU_AU": [0.4, 0.3, 0.2, 0.5, 0.4],
                "Zamber": [-0.7, -0.72, -0.71, -0.69, -0.7],
                "ZMonument": [1.2, 0.2, -0.4, -0.4, -0.4],
            }
        )
        result = infer_slot_contract_hypothesis(
            question="Which social capital value stayed low and showed low fluctuation in the younger bronze age (2300-2000 BCE)?",
            domain_context="Social Capital consists of Copper and Gold, Amber, Monument Count.",
            df=df,
            column_descriptions={
                "ZBeil": "Z values for axes",
                "ZCU_AU": "Z values for Copper and Gold",
                "Zamber": "Z values for Amber",
                "ZMonument": "Z values for Monument Count",
            },
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.slots["variable"], "ZMonument")
        self.assertIn("monument", result.hypothesis.lower())

    def test_pca_observed_component_closes_outlier_slots(self):
        import pandas as pd

        from mars.skills import infer_slot_contract_hypothesis

        df = pd.DataFrame(
            {
                "CE": [-4000, -3900, -3800, -3700, -3600, -3500],
                "Belau_PC1": [-2.0, -1.8, -1.6, -1.5, -1.4, 0.8],
                "Woserin_PC1": [-1.5, -1.3, -1.2, -1.0, -0.8, 1.1],
            }
        )
        result = infer_slot_contract_hypothesis(
            question=(
                "In the PCA analysis during the Early Neolithic period (4000-3500 BCE), "
                "what distinguishes the time slice around 3500 BCE from the general trend?"
            ),
            domain_context="PCA score columns measure first principal component values.",
            df=df,
            column_descriptions={
                "Belau_PC1": "PC1 of principal components",
                "Woserin_PC1": "PC1 of principal components",
            },
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.slots["event"], "pca_component")
        self.assertIn("PC1", result.hypothesis)
        self.assertIn("3500 BCE", result.hypothesis)
        self.assertIn("outlier", result.hypothesis.lower())

    def test_pca_loadings_close_social_capital_component_slots(self):
        import pandas as pd

        from mars.skills import infer_slot_contract_hypothesis

        df = pd.DataFrame(
            {
                "BCE": [4000, 3900, 3800, 3700, 3600, 3500],
                "ZMonument": [-2.0, -1.8, -1.6, -1.4, -1.2, -1.0],
                "ZCU_AU": [-1.9, -1.7, -1.5, -1.3, -1.1, -0.9],
                "Zamber": [-1.7, -1.6, -1.4, -1.2, -1.0, -0.8],
                "ZDolch": [1.0, 0.8, 0.5, 0.2, -0.1, -0.4],
                "ZBeil": [0.9, 0.7, 0.4, 0.1, -0.2, -0.5],
            }
        )
        result = infer_slot_contract_hypothesis(
            question=(
                "How are the elements of social capital, specifically the number of monuments, "
                "copper/gold, and amber, characterized in terms of PC1 and PC2 in the PCA?"
            ),
            domain_context="Social Capital consists of Copper and Gold, Amber, Monument Count.",
            df=df,
            column_descriptions={
                "ZMonument": "Z values for Monument Count",
                "ZCU_AU": "Z values for Copper and Gold",
                "Zamber": "Z values for Amber",
                "ZDolch": "Z values for Daggers",
                "ZBeil": "Z values for Axes",
            },
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.slots["event"], "pca_component")
        self.assertEqual(result.slots["group"], "social")
        self.assertIn("Social capital", result.hypothesis)
        self.assertIn("PC1", result.hypothesis)

    def test_window_relationship_uses_multiple_question_roles(self):
        import pandas as pd

        from mars.skills import infer_slot_contract_hypothesis

        df = pd.DataFrame(
            {
                "BCE": [3600, 3400, 3200, 3000, 2800],
                "Zamber": [-1.0, 1.2, 1.4, 1.1, -0.8],
                "ZMonument": [-0.8, 1.0, 1.3, 1.2, -0.7],
                "Zhausgr": [1.0, 0.5, 0.2, -0.2, 0.8],
            }
        )
        result = infer_slot_contract_hypothesis(
            question="What is the relationship of amber finds and number of monuments with house sizes between 3400-3000 BCE?",
            domain_context="Amber finds, monuments, and house sizes are forms of capital.",
            df=df,
            column_descriptions={
                "Zamber": "Z values for Amber",
                "ZMonument": "Z values for Monument Count",
                "Zhausgr": "Z values for House Size",
            },
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.slots["event"], "window_relationship")
        self.assertIn("amber", result.hypothesis.lower())
        self.assertIn("monuments", result.hypothesis.lower())
        self.assertIn("house", result.hypothesis.lower())

    def test_association_closes_source_and_target_slots(self):
        import pandas as pd

        from mars.skills import infer_slot_contract_hypothesis

        df = pd.DataFrame(
            {
                "Adjusted savings: education expenditure": [1, 2, 3, 4, 5, 6],
                "Labor force participation rate": [10, 12, 14, 16, 18, 20],
                "GNI per capita": [100, 110, 130, 150, 170, 190],
                "Exports growth": [2, 1, 3, 2, 4, 3],
            }
        )
        result = infer_slot_contract_hypothesis(
            question="How does increased education expenditure influence human capital and economic output?",
            domain_context="Labor force is a proxy for human capital. GNI per capita represents economic output.",
            df=df,
            column_descriptions={
                "Adjusted savings: education expenditure": "government education investment",
                "Labor force participation rate": "labor force proxy for human capital",
                "GNI per capita": "economic output and income",
                "Exports growth": "exports",
            },
        )

        self.assertIsNotNone(result)
        assert result is not None
        text = f"{result.hypothesis} {result.workflow}".lower()
        self.assertIn("education", text)
        self.assertTrue("gni" in text or "economic output" in text)


if __name__ == "__main__":
    unittest.main()
