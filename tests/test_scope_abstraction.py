import unittest


class ScopeAbstractionTests(unittest.TestCase):
    def test_broadens_local_group_when_relation_is_stable(self):
        from mars.skills import MeasuredPairGroup, induce_scope_abstraction

        scope = induce_scope_abstraction(
            question="In Experimental Economics, what is the average effect estimate as compared to that in replication studies?",
            requested_groups=("Experimental Economics",),
            rows=[
                MeasuredPairGroup("Experimental Economics", 0.57, 0.31, 18),
                MeasuredPairGroup("Psychology", 0.50, 0.24, 68),
            ],
        )
        self.assertIsNotNone(scope)
        assert scope is not None
        self.assertEqual(scope.mode, "local_plus_global")
        self.assertIn("Experimental Economics", scope.groups)
        self.assertIn("Psychology", scope.groups)

    def test_keeps_lookup_wording_local(self):
        from mars.skills import MeasuredPairGroup, induce_scope_abstraction

        scope = induce_scope_abstraction(
            question="In Experimental Economics, what is the average effect estimate compared with replication studies?",
            requested_groups=("Experimental Economics",),
            rows=[
                MeasuredPairGroup("Experimental Economics", 0.57, 0.31, 18),
                MeasuredPairGroup("Psychology", 0.50, 0.24, 68),
            ],
        )
        self.assertIsNone(scope)


if __name__ == "__main__":
    unittest.main()
