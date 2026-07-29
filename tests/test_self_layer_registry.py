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

    def test_probe_reuses_source_hash_when_storage_header_is_present(self):
        with tempfile.TemporaryDirectory() as td:
            registry = SelfLayerRegistry(td)
            code = (
                "def build_layer(context: dict) -> dict:\n"
                "    return {'program_sources': [], 'signals': ['ok']}\n"
            )
            first = registry.promote(
                namespace="universal",
                name="stable",
                layer_type="signal",
                code=code,
                contract={"input": "context"},
                description="stable",
                score={"gain": 0.4, "loss_mean": 0.0, "validation_key": "a"},
            )
            assert first is not None
            stored = Path(first.path).read_text(encoding="utf-8")
            second = registry.promote(
                namespace="universal",
                name="stable",
                layer_type="signal",
                code=stored,
                contract={"input": "context"},
                description="stable",
                score={"gain": 0.4, "loss_mean": 0.0, "validation_key": "b"},
            )
            assert second is not None
            self.assertEqual(first.source_hash, second.source_hash)

    def test_quarantined_layer_requires_independent_probe_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            registry = SelfLayerRegistry(td)
            code = (
                "def build_layer(context: dict) -> dict:\n"
                "    return {'program_sources': [], 'signals': ['ok']}\n"
            )
            for key, phase in (("source", "induction"), ("probe_a", "promotion_probe")):
                self.assertIsNotNone(
                    registry.promote(
                        namespace="universal",
                        name="identity",
                        layer_type="signal",
                        code=code,
                        contract={"input": "context"},
                        description="identity",
                        score={
                            "gain": 0.9,
                            "loss_mean": 0.0,
                            "validation_key": key,
                            "evidence_phase": phase,
                        },
                        min_gain=0.0,
                        max_loss_mean=1.0,
                    )
                )
            self.assertEqual(
                registry.refresh_trusted_manifest(
                    "universal", min_probe_validation_keys=2
                ),
                [],
            )
            self.assertIsNotNone(
                registry.promote(
                    namespace="universal",
                    name="identity",
                    layer_type="signal",
                    code=code,
                    contract={"input": "context"},
                    description="identity",
                    score={
                        "gain": 0.9,
                        "loss_mean": 0.0,
                        "validation_key": "probe_b",
                        "evidence_phase": "promotion_probe",
                    },
                    min_gain=0.0,
                    max_loss_mean=1.0,
                )
            )
            trusted = registry.refresh_trusted_manifest(
                "universal", min_probe_validation_keys=2
            )
            self.assertEqual(len(trusted), 1)
            self.assertEqual(set(trusted[0].probe_validation_keys), {"probe_a", "probe_b"})


if __name__ == "__main__":
    unittest.main()
