import tempfile
import unittest

from mars.induction.universal_cpi import UniversalCPI
from mars.skills.residual_class_ledger import ResidualClassLedger


class _SignatureAdapter:
    def __init__(self, signature):
        self._signature = signature

    def signature_hint(self):
        return self._signature


class ResidualClassLedgerTests(unittest.TestCase):
    def test_groups_without_recording_task_contents_and_claims_once(self):
        with tempfile.TemporaryDirectory() as td:
            ledger = ResidualClassLedger(td)
            fingerprint = {"input_type": "dict", "target_type": "number", "loss_band": "high"}
            class_id, support = ledger.observe(validation_key="train:a", fingerprint=fingerprint)
            self.assertEqual(support, 1)
            same_id, support = ledger.observe(validation_key="train:b", fingerprint=fingerprint)
            self.assertEqual(class_id, same_id)
            self.assertEqual(support, 2)
            self.assertTrue(ledger.claim(class_id))
            self.assertFalse(ledger.claim(class_id))

    def test_groups_same_typed_interface_despite_loss_band(self):
        with tempfile.TemporaryDirectory() as td:
            ledger = ResidualClassLedger(td)
            common = {
                "input_type": "dict",
                "target_type": "float",
                "signature": "slot slot(slot) -> slot",
            }
            first, _ = ledger.observe(
                validation_key="gravity",
                fingerprint={**common, "loss_band": "low", "support_bucket": "medium"},
            )
            second, support = ledger.observe(
                validation_key="coulomb",
                fingerprint={**common, "loss_band": "mixed", "support_bucket": "large"},
            )
            self.assertEqual(first, second)
            self.assertEqual(support, 2)

    def test_residual_signature_ignores_task_local_field_comments(self):
        engine = UniversalCPI(
            model="openai/gpt-4o-mini",
            n_proposals=1,
            max_rounds=0,
        )
        first = engine._residual_fingerprint(
            _SignatureAdapter("discovered_law(*args)  # mass1, mass2, distance"),
            [],
            None,
        )
        second = engine._residual_fingerprint(
            _SignatureAdapter("discovered_law(*args)  # omega, temperature"),
            [],
            None,
        )
        self.assertEqual(first["signature"], second["signature"])


if __name__ == "__main__":
    unittest.main()
