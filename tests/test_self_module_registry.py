import tempfile
import unittest
from pathlib import Path

from mars.skills.self_module_registry import (
    SelfModuleRegistry,
    validate_self_module_source,
)


class SelfModuleRegistryTests(unittest.TestCase):
    def test_safety_rejects_dangerous_source(self):
        ok, reason = validate_self_module_source("def f():\n    return open('/tmp/x').read()\n")
        self.assertFalse(ok)
        self.assertIn("unsafe", reason)

    def test_promote_requires_score_gate_and_writes_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            registry = SelfModuleRegistry(td)
            code = "def rule(current: str, context: dict) -> str:\n    return current\n"

            weak = registry.promote(
                namespace="uh_seq",
                name="identity",
                code=code,
                description="identity",
                score={"exact_rate": 0.1, "loss_mean": 0.5},
            )
            self.assertIsNone(weak)

            good = registry.promote(
                namespace="uh_seq",
                name="identity",
                code=code,
                description="identity",
                score={"exact_rate": 0.8, "loss_mean": 0.0},
            )
            self.assertIsNotNone(good)
            assert good is not None
            self.assertTrue(Path(good.path).exists())
            records = registry.load("uh_seq")
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].description, "identity")

    def test_trusted_manifest_requires_multiple_validation_keys(self):
        with tempfile.TemporaryDirectory() as td:
            registry = SelfModuleRegistry(td)
            code = "def rule(current: str, context: dict) -> str:\n    return current\n"

            one = registry.promote(
                namespace="uh_seq",
                name="identity",
                code=code,
                description="identity",
                score={"exact_rate": 0.8, "loss_mean": 0.0, "validation_key": "seed_a"},
            )
            self.assertIsNotNone(one)
            self.assertEqual(registry.refresh_trusted_manifest("uh_seq"), [])

            two = registry.promote(
                namespace="uh_seq",
                name="identity",
                code=code,
                description="identity",
                score={"exact_rate": 0.7, "loss_mean": 0.1, "validation_key": "seed_b"},
            )
            self.assertIsNotNone(two)

            trusted = registry.refresh_trusted_manifest("uh_seq")
            self.assertEqual(len(trusted), 1)
            self.assertEqual(set(trusted[0].validation_keys), {"seed_a", "seed_b"})
            self.assertEqual(len(registry.load_trusted("uh_seq")), 1)


if __name__ == "__main__":
    unittest.main()
