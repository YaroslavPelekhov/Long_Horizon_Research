import unittest

from mars.runners.summarize_newton_baseline_failures import classify_law


class NewtonBaselineFailureAuditTests(unittest.TestCase):
    def test_classifies_exact_before_form_checks(self):
        self.assertEqual(classify_law("", exact=True), "exact")

    def test_classifies_empty_and_invalid_sources(self):
        self.assertEqual(classify_law("", exact=False), "empty")
        self.assertEqual(classify_law("def broken(", exact=False), "invalid_syntax")

    def test_distinguishes_constant_from_variable_dependent_law(self):
        constant = "def discovered_law(x):\n    return 0"
        dependent = "def discovered_law(x):\n    return 2 * x"
        self.assertEqual(classify_law(constant, exact=False), "constant_only")
        self.assertEqual(
            classify_law(dependent, exact=False), "valid_but_incorrect"
        )


if __name__ == "__main__":
    unittest.main()
