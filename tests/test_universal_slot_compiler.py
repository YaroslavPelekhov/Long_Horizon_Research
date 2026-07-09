import unittest


class UniversalSlotCompilerTests(unittest.TestCase):
    def test_question_form_inducer_classifies_measurable_forms(self):
        from mars.skills import infer_question_form

        form = infer_question_form(
            "Which variables have a positive relationship quantified by a coefficient of 0.22?"
        )

        self.assertIsNotNone(form)
        assert form is not None
        self.assertEqual(form.name, "stated_coefficient_relationship")
        self.assertEqual(form.constants["coefficient"], 0.22)

    def test_discovery_table_backend_closes_executable_slots(self):
        import pandas as pd

        from mars.skills import infer_universal_slot_hypothesis

        df = pd.DataFrame(
            {
                "discipline": ["Economics", "Economics", "Cognitive", "Cognitive"],
                "fiso": [0.6, 0.5, 0.7, 0.5],
                "fisr": [0.3, 0.2, 0.4, 0.3],
            }
        )
        result = infer_universal_slot_hypothesis(
            task_text="For which domains do the effect size estimates tend to be larger in original studies compared to replication studies?",
            interface_kind="discovery_table",
            data=df,
            schema={
                "discipline": "Discipline or domain",
                "fiso": "Effect estimate of original study transformed to Fisher-z scale",
                "fisr": "Effect estimate of replication study transformed to Fisher-z scale",
            },
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.slots["backend"], "discovery_table")
        self.assertIn("UniversalSlotCompiler", result.workflow)
        self.assertIn("Experimental Economics", result.hypothesis)

    def test_discovery_table_closes_paired_power_means(self):
        import pandas as pd

        from mars.skills import infer_universal_slot_hypothesis

        df = pd.DataFrame(
            {
                "project.x": ["Experimental Economics", "Experimental Economics", "Psychology", "Psychology"],
                "power.o": [0.8, 0.9, 0.7, 1.0],
                "power_planned.r": [0.95, 0.91, 0.9, 0.96],
            }
        )
        result = infer_universal_slot_hypothesis(
            task_text="In Experimental Economics, what were the average observed power in original studies and the planned power in replication studies?",
            interface_kind="discovery_table",
            data=df,
            schema={
                "project.x": "The replication project that the study was on",
                "power.o": "Post hoc power based on original effect size",
                "power_planned.r": "Planned power of the replication based on planned N and original ES",
            },
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("0.85", result.hypothesis)
        self.assertIn("0.93", result.hypothesis)

    def test_discovery_table_closes_stated_coefficient_pair(self):
        import pandas as pd

        from mars.skills import infer_universal_slot_hypothesis

        df = pd.DataFrame(
            {
                "urban_land_use": [0, 1, 2, 3, 4, 5, 6, 7],
                "gardening_proportion": [0, 0.1, 0.2, 0.5, 0.9, 1.2, 1.7, 2.0],
                "noise": [7, 1, 7, 1, 7, 1, 7, 1],
            }
        )
        result = infer_universal_slot_hypothesis(
            task_text="What are the variables between which a positive relationship is quantified by a coefficient of 0.95?",
            interface_kind="discovery_table",
            data=df,
            schema={
                "urban_land_use": "degree of urban land use",
                "gardening_proportion": "proportion of gardening-introduced non-native plants",
                "noise": "irrelevant numeric column",
            },
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("urban land use", result.hypothesis)
        self.assertIn("gardening-introduced non-native plants", result.hypothesis)

    def test_discovery_table_closes_prompted_survey_items(self):
        import pandas as pd

        from mars.skills import infer_universal_slot_hypothesis

        df = pd.DataFrame(
            {
                "Q11_ML_NFRs_System_Performance": ["quoted"] * 35 + ["not quoted"] * 65,
                "Q11_ML_NFRs_System_Usability": ["quoted"] * 25 + ["not quoted"] * 75,
                "Q11_ML_NFRs_Model_Reliability": ["quoted"] * 80 + ["not quoted"] * 20,
            }
        )
        result = infer_universal_slot_hypothesis(
            task_text=(
                "Which two Non-Functional Requirements regarding the whole system are considered important "
                "in ML-enabled system projects, with 35.0% (95% CI [34, 36]) and 25.0% "
                "(95% CI [24, 26]) of respondents indicating so, respectively?"
            ),
            interface_kind="discovery_table",
            data=df,
            schema={
                "Q11_ML_NFRs_System_Performance": "Degree to which system performance is considered as a non-functional requirement",
                "Q11_ML_NFRs_System_Usability": "Degree to which system usability is considered as a non-functional requirement",
                "Q11_ML_NFRs_Model_Reliability": "Degree to which model reliability is considered as a non-functional requirement",
            },
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("System Performance", result.hypothesis)
        self.assertIn("Usability", result.hypothesis)

    def test_discovery_table_falls_back_to_generic_categorical_measurement(self):
        import pandas as pd

        from mars.skills import infer_universal_slot_hypothesis

        df = pd.DataFrame(
            {
                "domain": ["Experimental Economics"] * 4 + ["Psychology"] * 2,
                "subjects.r": ["students", "students", "students", "students", "students", "community"],
            }
        )
        result = infer_universal_slot_hypothesis(
            task_text="What type of subjects were used in all replication studies in Experimental Economics?",
            interface_kind="discovery_table",
            data=df,
            schema={
                "domain": "Scientific domain",
                "subjects.r": "Type of subjects used in the replication study",
            },
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.slots["backend"], "discovery_table")
        self.assertIn("student subjects", result.hypothesis)
        self.assertIn("generic_categorical_measurement", result.workflow)

    def test_universal_plans_cover_four_benchmark_artifacts(self):
        from mars.skills import compile_universal_slot_plan

        law = compile_universal_slot_plan(
            task_text="Infer a physical law from observations.",
            interface_kind="newton_law",
            metadata={"variables": ["m1", "m2", "r"], "target": "F"},
        )
        program = compile_universal_slot_plan(
            task_text="Infer a hidden transformation from input-output traces.",
            interface_kind="ultrahorizon_program",
        )
        workflow = compile_universal_slot_plan(
            task_text="Complete a scientific tool-use task and validate the final artifact.",
            interface_kind="science_workflow",
        )

        self.assertIsNotNone(law)
        self.assertIsNotNone(program)
        self.assertIsNotNone(workflow)
        assert law is not None and program is not None and workflow is not None
        self.assertEqual(law.artifact_kind, "scientific_law")
        self.assertIn("constants", law.slots)
        self.assertEqual(program.artifact_kind, "program")
        self.assertIn("composition", program.slots)
        self.assertEqual(workflow.artifact_kind, "workflow")
        self.assertIn("validation_checks", workflow.slots)

    def test_multitable_worldbank_closes_cross_file_panel_slots(self):
        import pandas as pd

        from mars.skills import infer_universal_slot_hypothesis

        edu = pd.DataFrame(
            {
                "Country Group": ["Sub-Saharan Africa", "Lower middle income"],
                "1975 [YR1975]": [1.0, 2.0],
                "1976 [YR1976]": [2.0, 3.0],
                "1977 [YR1977]": [3.0, 4.0],
            }
        )
        gni = pd.DataFrame(
            {
                "Country Group": ["Sub-Saharan Africa", "Lower middle income"],
                "1975 [YR1975]": [10.0, 20.0],
                "1976 [YR1976]": [20.0, 30.0],
                "1977 [YR1977]": [30.0, 40.0],
            }
        )
        result = infer_universal_slot_hypothesis(
            task_text="In what regions does increased education spending positively impact per capita GDP?",
            interface_kind="discovery_multitable",
            data={
                "Adjusted_savings_education_expenditure_percentage_of_GNI.csv": edu,
                "GNI_per_capita_constant_2015_USdollar.csv": gni,
            },
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.slots["backend"], "discovery_multitable")
        self.assertIn("developing countries", result.hypothesis)
        self.assertIn("answer_slot_multitable_worldbank", result.evidence)

    def test_multitable_panel_impact_can_use_lagged_effect(self):
        import pandas as pd

        from mars.skills import infer_universal_slot_hypothesis

        edu = pd.DataFrame(
            {
                "Country Group": ["Sub-Saharan Africa", "Lower middle income"],
                "1975 [YR1975]": [1, 1],
                "1976 [YR1976]": [3, 3],
                "1977 [YR1977]": [6, 6],
                "1978 [YR1978]": [9, 9],
                "1979 [YR1979]": [7, 7],
                "1980 [YR1980]": [9, 9],
            }
        )
        gni = pd.DataFrame(
            {
                "Country Group": ["Sub-Saharan Africa", "Lower middle income"],
                "1975 [YR1975]": [43, 43],
                "1976 [YR1976]": [33, 33],
                "1977 [YR1977]": [40, 40],
                "1978 [YR1978]": [44, 44],
                "1979 [YR1979]": [35, 35],
                "1980 [YR1980]": [2, 2],
            }
        )

        result = infer_universal_slot_hypothesis(
            task_text="In what regions does increased education spending positively impact per capita GDP?",
            interface_kind="discovery_multitable",
            data={
                "Adjusted_savings_education_expenditure_percentage_of_GNI.csv": edu,
                "GNI_per_capita_constant_2015_USdollar.csv": gni,
            },
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("positive impact", result.hypothesis)
        self.assertIn("developing countries", result.hypothesis)

    def test_multitable_meta_regression_prefers_human_domain_table(self):
        import pandas as pd

        from mars.skills import infer_universal_slot_hypothesis

        coded = pd.DataFrame(
            {
                "project": ["ee", "ee", "rpp", "rpp"],
                "effect_size.o": [0.4, 0.5, 0.3, 0.4],
                "effect_size.r": [0.2, 0.3, 0.1, 0.2],
            }
        )
        human = pd.DataFrame(
            {
                "project": ["Psychology", "Psychology", "Experimental Economics", "Experimental Economics", "Social Sciences", "Social Sciences"],
                "fiso": [0.5, 0.5, 0.57, 0.57, 0.6, 0.6],
                "fisr": [0.24, 0.24, 0.31, 0.31, 0.3, 0.3],
            }
        )
        result = infer_universal_slot_hypothesis(
            task_text="For which domains do the effect size estimates tend to be larger in original studies compared to replication studies?",
            interface_kind="discovery_multitable",
            data={
                "meta-regression_study_data.csv": coded,
                "meta-regression_replication_success_data.csv": human,
            },
            schema={
                "meta-regression_study_data.csv": {
                    "project": "project code",
                    "effect_size.o": "Standardized effect size of original paper",
                    "effect_size.r": "Standardized effect size of replication",
                },
                "meta-regression_replication_success_data.csv": {
                    "project": "Name of replication project",
                    "fiso": "Effect estimate of original study transformed to Fisher-z scale",
                    "fisr": "Effect estimate of replication study transformed to Fisher-z scale",
                },
            },
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertIn("Experimental Economics", result.hypothesis)
        self.assertIn("Psychology", result.hypothesis)
        self.assertNotIn("Social Sciences", result.hypothesis)


if __name__ == "__main__":
    unittest.main()
