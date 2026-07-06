import unittest


class AbstractionPosteriorTests(unittest.TestCase):
    def test_induces_semantic_habitat_contrast(self):
        from mars.skills import MeasuredGroup, induce_contrast_abstraction

        abstraction = induce_contrast_abstraction(
            question="How does prevalence vary based on habitat type?",
            groups=[
                MeasuredGroup("Urban", 3, 20, 15.0),
                MeasuredGroup("Croplands", 2, 20, 10.0),
                MeasuredGroup("Broad-leaved forests", 8, 10, 80.0),
                MeasuredGroup("Wetlands", 6, 10, 60.0),
            ],
        )
        self.assertIsNotNone(abstraction)
        assert abstraction is not None
        self.assertEqual(abstraction.name, "human_modified_vs_natural")
        self.assertEqual(abstraction.direction, "lower")
        self.assertIn("Urban", abstraction.left_groups)
        self.assertIn("Wetlands", abstraction.right_groups)

    def test_metric_gap_fallback_when_no_semantic_axis_exists(self):
        from mars.skills import MeasuredGroup, induce_contrast_abstraction

        abstraction = induce_contrast_abstraction(
            question="How does the proportion vary across categories?",
            groups=[
                MeasuredGroup("A", 9, 10, 90.0),
                MeasuredGroup("B", 8, 10, 80.0),
                MeasuredGroup("C", 2, 10, 20.0),
                MeasuredGroup("D", 1, 10, 10.0),
            ],
        )
        self.assertIsNotNone(abstraction)
        assert abstraction is not None
        self.assertEqual(abstraction.name, "metric_gap_partition")
        self.assertGreater(abstraction.left_value, abstraction.right_value)


if __name__ == "__main__":
    unittest.main()
