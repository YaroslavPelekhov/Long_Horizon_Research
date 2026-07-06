import unittest

from mars.induction.universal_cpi import Observation
from mars.skills import profile_observations, profile_prompt


class InterfaceProfilerTests(unittest.TestCase):
    def test_table_profile_separates_axes_from_variables(self):
        import pandas as pd

        df = pd.DataFrame(
            {
                "1975 [YR1975]": [1, 2, 3],
                "Year": [1975, 1976, 1977],
                "education expenditure": [0.1, 0.2, 0.3],
                "GNI per capita": [10, 12, 15],
            }
        )
        profile = profile_observations([Observation(inputs=df, target=None)])

        self.assertEqual(profile.input_family, "table")
        self.assertIn("1975 [YR1975]", profile.candidate_axes)
        self.assertIn("Year", profile.candidate_axes)
        self.assertIn("education expenditure", profile.candidate_variables)
        self.assertNotIn("1975 [YR1975]", profile.candidate_variables)
        self.assertIn("axis_variable_separation", profile.operator_families)
        self.assertIn("axis_columns_are_context_not_scientific_variables", profile.warnings)

    def test_mapping_numeric_profile_suggests_formula_search(self):
        obs = [
            Observation(inputs={"m1": 2.0, "m2": 3.0, "r": 2.0}, target=1.5),
            Observation(inputs={"m1": 4.0, "m2": 5.0, "r": 2.0}, target=5.0),
        ]
        profile = profile_observations(obs)

        self.assertEqual(profile.input_family, "mapping")
        self.assertEqual(profile.target_family, "number")
        self.assertIn("power_law_probe", profile.operator_families)
        self.assertIn("m1", profile.candidate_variables)

    def test_string_profile_suggests_transform_probe(self):
        obs = [
            Observation(inputs="AB", target="BA", context={"step_number": 1}),
            Observation(inputs="CD", target="DC", context={"step_number": 2}),
        ]
        profile = profile_observations(obs)
        prompt = profile_prompt(obs)

        self.assertEqual(profile.input_family, "string")
        self.assertIn("string_transform_probe", profile.operator_families)
        self.assertIn("step_condition_branch_probe", profile.operator_families)
        self.assertIn("UNIVERSAL INTERFACE PROFILE", prompt)


if __name__ == "__main__":
    unittest.main()
