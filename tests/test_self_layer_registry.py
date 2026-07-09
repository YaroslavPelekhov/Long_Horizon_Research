import tempfile
import unittest
from pathlib import Path

from mars.skills.self_layer_registry import (
    SelfLayerRegistry,
    validate_self_layer_source,
)


class SelfLayerRegistryTests(unittest.TestCase):
    def test_safety_rejects_dangerous_layer(self):
        code = "def build_layer(context: dict) -> dict:\n    return open('/tmp/x').read()\n"
        ok, reason = validate_self_layer_source(code)
        self.assertFalse(ok)
        self.assertIn("unsafe", reason)

    def test_promote_requires_gate_and_writes_manifest(self):
        with tempfile.TemporaryDirectory() as td:
            registry = SelfLayerRegistry(td)
            code = (
                "def build_layer(context: dict) -> dict:\n"
                "    return {'program_sources': [], 'signals': ['ok']}\n"
            )

            weak = registry.promote(
                namespace="universal",
                name="empty",
                layer_type="signal",
                code=code,
                contract={"input": "context"},
                description="empty layer",
                score={"gain": 0.0, "loss_mean": 0.8},
            )
            self.assertIsNone(weak)

            good = registry.promote(
                namespace="universal",
                name="empty",
                layer_type="signal",
                code=code,
                contract={"input": "context"},
                description="empty layer",
                score={"gain": 0.3, "loss_mean": 0.0, "validation_key": "a"},
            )
            self.assertIsNotNone(good)
            assert good is not None
            self.assertTrue(Path(good.path).exists())
            self.assertEqual(len(registry.load("universal")), 1)

    def test_trusted_manifest_requires_multiple_validation_keys(self):
        with tempfile.TemporaryDirectory() as td:
            registry = SelfLayerRegistry(td)
            code = (
                "def build_layer(context: dict) -> dict:\n"
                "    return {'program_sources': [], 'signals': ['ok']}\n"
            )
            for key in ("a", "b"):
                rec = registry.promote(
                    namespace="universal",
                    name="empty",
                    layer_type="signal",
                    code=code,
                    contract={"input": "context"},
                    description="empty layer",
                    score={"gain": 0.4, "loss_mean": 0.0, "validation_key": key},
                )
                self.assertIsNotNone(rec)

            trusted = registry.refresh_trusted_manifest("universal")
            self.assertEqual(len(trusted), 1)
            self.assertEqual(set(trusted[0].validation_keys), {"a", "b"})
            self.assertEqual(len(registry.load_trusted("universal")), 1)


if __name__ == "__main__":
    unittest.main()
