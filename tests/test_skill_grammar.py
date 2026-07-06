import unittest

from mars.skills import ResidualCase, SkillGrammar, anti_unify


class SkillGrammarTests(unittest.TestCase):
    def test_anti_unify_threshold_skills(self):
        result = anti_unify(
            [
                ("threshold_split", "energy", "<=", 13.5),
                ("threshold_split", "age", "<=", 13.5),
                ("threshold_split", "temperature", "<=", 13.5),
            ]
        )
        self.assertEqual(result.schema, ("threshold_split", "?v0", "<=", 13.5))
        self.assertGreater(result.compression_gain, 0)

    def test_skill_grammar_birth_and_consolidation(self):
        def batch(feature):
            return [
                ResidualCase({feature: 2}, "old", "new", 1.0),
                ResidualCase({feature: 5}, "old", "new", 1.0),
                ResidualCase({feature: 20}, "old", "old", 0.0),
                ResidualCase({feature: 25}, "old", "old", 0.0),
            ]

        grammar = SkillGrammar(min_gain=0.1)
        report = grammar.induce(
            [
                ("energy_skill", batch("energy")),
                ("age_skill", batch("age")),
            ]
        )
        self.assertEqual([skill.name for skill in report.born], ["energy_skill", "age_skill"])
        self.assertTrue(report.consolidated)
        self.assertEqual(report.consolidated[0].schema[0], "threshold_split")

    def test_interaction_skill_ignores_constant_numeric_features(self):
        cases = [
            ResidualCase({"bias": 1, "x": 1, "y": 2}, 1, 2, 1.0),
            ResidualCase({"bias": 1, "x": 2, "y": 3}, 2, 6, 1.0),
            ResidualCase({"bias": 1, "x": 3, "y": 5}, 3, 15, 1.0),
        ]
        grammar = SkillGrammar()
        skill = grammar.birth("pair_skill", cases)
        self.assertIsNotNone(skill)
        self.assertEqual(skill.schema, ("interaction_product", "x", "y"))


if __name__ == "__main__":
    unittest.main()
