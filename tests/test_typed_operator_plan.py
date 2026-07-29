import unittest

from mars.skills.typed_operator_plan import (
    admissible_families,
    compile_typed_operator_plan,
)


class TypedOperatorPlanTests(unittest.TestCase):
    def test_type_gate_exposes_only_compatible_families(self):
        self.assertEqual(
            admissible_families(
                input_type="DataFrame",
                target_type="str",
                signature_hint="def analyze(df) -> dict",
            ),
            ["table_numeric_association"],
        )
        self.assertEqual(
            admissible_families(
                input_type="dict",
                target_type="float",
                signature_hint="def law(inputs: dict) -> float",
            ),
            ["positive_monomial_lattice", "positive_additive_lattice"],
        )

    def test_compiler_rejects_incompatible_plan(self):
        compiled = compile_typed_operator_plan(
            {"family": "positive_monomial_lattice"},
            input_type="DataFrame",
            target_type="str",
            signature_hint="def analyze(df) -> dict",
        )
        self.assertIsNone(compiled)

    def test_compiler_emits_dynamic_table_layer(self):
        compiled = compile_typed_operator_plan(
            {"family": "table_numeric_association", "description": "dynamic association"},
            input_type="DataFrame",
            target_type="str",
            signature_hint="def analyze(df) -> dict",
        )
        self.assertIsNotNone(compiled)
        assert compiled is not None
        self.assertIn("df.select_dtypes", compiled["code"])
        self.assertNotIn("source-task", compiled["code"])

    def test_compiler_emits_generic_additive_numeric_layer(self):
        compiled = compile_typed_operator_plan(
            {"family": "positive_additive_lattice", "description": "numeric additive control"},
            input_type="dict",
            target_type="float",
            signature_hint="def law(inputs: dict) -> float",
        )
        self.assertIsNotNone(compiled)
        assert compiled is not None
        self.assertIn("typed_additive_coordinate", compiled["code"])
        self.assertNotIn("gravity", compiled["code"].lower())


if __name__ == "__main__":
    unittest.main()
