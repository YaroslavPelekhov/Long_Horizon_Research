import unittest
from unittest.mock import patch

from mars.skills.program_induction_ledger import induce_trace_rule_report


class ProgramInductionLedgerTests(unittest.TestCase):
    def test_closes_typed_trace_slots_before_autonomous_cpi(self):
        history = [
            {
                "result": {
                    "success": True,
                    "main_input": "ABCDE",
                    "vice_input": "EDCBA",
                    "step_number": 1,
                    "transformations": [
                        {"step": 0, "sequence": ""},
                        {"step": 1, "sequence": "AEBDCCDBEA"},
                        {"step": 2, "sequence": "EEEEE"},
                        {"step": 3, "sequence": "EDCDE"},
                        {"step": 4, "sequence": "EDCDE"},
                        {"step": 5, "sequence": "AABBCCDDEE"},
                    ],
                }
            },
            {
                "result": {
                    "success": True,
                    "main_input": "AABCD",
                    "vice_input": "DDECB",
                    "step_number": 2,
                    "transformations": [
                        {"step": 0, "sequence": ""},
                        {"step": 1, "sequence": "ADADBECCDB"},
                        {"step": 2, "sequence": "DDFEE"},
                        {"step": 3, "sequence": "DDECD"},
                        {"step": 4, "sequence": "DDECD"},
                        {"step": 5, "sequence": "AABBCCDDDE"},
                    ],
                }
            },
        ]

        report = induce_trace_rule_report(history)

        self.assertIsNotNone(report)
        assert report is not None
        self.assertEqual(report.mode, "typed_trace_slots")
        self.assertIn("rule_1", report.artifact)
        self.assertIn("Alternate characters", report.artifact)
        self.assertIn("rule_5", report.artifact)
        self.assertIn("Combine main and vice strings then sort alphabetically", report.artifact)
        self.assertIn("sort current string if no main/vice", report.artifact)
        self.assertGreaterEqual(len(report.matched_rules), 5)

    def test_induces_sequence_trace_rules(self):
        history = [
            {
                "result": {
                    "main_input": "ABCDE",
                    "vice_input": "EDCBA",
                    "step_number": 1,
                    "transformations": [
                        {"step": 1, "sequence": "AEBDCCDBEA"},
                        {"step": 2, "sequence": "BFCEDDECFBBFCEDDECFB"},
                        {"step": 3, "sequence": "BFCEDDECFBBFCEDDECFBB"},
                        {"step": 4, "sequence": "BFCEDDECFBBFCEDDECFBB"},
                        {"step": 5, "sequence": "BFCEDDECFBBFCEDDECFBB"},
                    ],
                }
            },
            {
                "result": {
                    "main_input": "AABCD",
                    "vice_input": "DDECB",
                    "step_number": 2,
                    "transformations": [
                        {"step": 1, "sequence": "DADAEBCCBD"},
                        {"step": 2, "sequence": "FDEEDGCFCFFCFCGDEEDF"},
                        {"step": 3, "sequence": "FDEEDGCFCFFCFCGDEEDFDD"},
                        {"step": 4, "sequence": "FEGHHGCFCFFCFCGDEEDFDD"},
                        {"step": 5, "sequence": "GEGHHGCGCGGCGCGDEEDGDD"},
                    ],
                }
            },
        ]
        with patch.dict(
            "os.environ",
            {
                "MARS_USE_AUTONOMOUS_INDUCTION": "0",
                "MARS_ALLOW_HAND_GRAMMAR_FALLBACK": "1",
            },
        ):
            report = induce_trace_rule_report(history)
        self.assertIsNotNone(report)
        assert report is not None
        self.assertIn("rule_1", report.artifact)
        self.assertIn("interleave", report.artifact)
        self.assertIn("rule_2", report.artifact)
        self.assertIn("reverse", report.artifact)
        self.assertIn("rule_3", report.artifact)
        self.assertIn("step_number", report.artifact)


if __name__ == "__main__":
    unittest.main()
