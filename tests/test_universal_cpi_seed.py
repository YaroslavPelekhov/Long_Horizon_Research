import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from mars.induction.universal_cpi import (
    CPIAdapter,
    HypothesisProgram,
    Observation,
    UniversalCPI,
    _sandbox_compile,
)
from mars.skills.typed_operator_plan import compile_typed_operator_plan
from mars.skills.self_layer_registry import SelfLayerRegistry


class SeedAdapter(CPIAdapter):
    name = "seed_test"

    def interface_description(self) -> str:
        return "Toy string task."

    def signature_hint(self) -> str:
        return "def rule(current: str, context: dict) -> str"

    def collect_observations(self):
        return [
            Observation(inputs="a", target="ax", context={}),
            Observation(inputs="b", target="bx", context={}),
            Observation(inputs="c", target="cx", context={}),
        ]

    def execute(self, program, obs):
        return program.fn(obs.inputs, obs.context)

    def loss(self, prediction, obs) -> float:
        return 0.0 if prediction == obs.target else 1.0

    def render_report(self, winners, observations) -> str:
        return winners[0][0].description if winners else ""

    def seed_program_sources(self):
        return [
            {
                "name": "append_x",
                "description": "append x",
                "code": "def rule(current: str, context: dict) -> str:\n    return current + 'x'\n",
            }
        ]


class CompositionSeedAdapter(SeedAdapter):
    def collect_observations(self):
        return [
            Observation(inputs="a", target="axy", context={}),
            Observation(inputs="b", target="bxy", context={}),
            Observation(inputs="c", target="cxy", context={}),
        ]

    def seed_program_sources(self):
        return [
            {
                "name": "append_x",
                "description": "append x",
                "code": "def rule(current: str, context: dict) -> str:\n    return current + 'x'\n",
            },
            {
                "name": "append_y",
                "description": "append y",
                "code": "def rule(current: str, context: dict) -> str:\n    return current + 'y'\n",
            },
        ]

    def compose_seed_programs(self) -> bool:
        return True


class BranchCompositionSeedAdapter(SeedAdapter):
    def collect_observations(self):
        return [
            Observation(inputs="a", target="ay", context={"step_number": 1}),
            Observation(inputs="b", target="bx", context={"step_number": 2}),
            Observation(inputs="c", target="cy", context={"step_number": 3}),
            Observation(inputs="d", target="dx", context={"step_number": 4}),
        ]

    def seed_program_sources(self):
        return [
            {
                "name": "append_x",
                "description": "append x",
                "code": "def rule(current: str, context: dict) -> str:\n    return current + 'x'\n",
            },
            {
                "name": "append_y",
                "description": "append y",
                "code": "def rule(current: str, context: dict) -> str:\n    return current + 'y'\n",
            },
        ]

    def branch_seed_programs(self) -> bool:
        return True

    def branch_predicate_sources(self, observations):
        return [
            {
                "name": "step_even",
                "expr": "int(context.get('step_number', 0)) % 2 == 0",
            }
        ]


class TypedPriorAdapter(SeedAdapter):
    def collect_observations(self):
        return [
            Observation(inputs="AB", target="CBBC", context={"step_number": 1}),
            Observation(inputs="BC", target="EDDE", context={"step_number": 2}),
            Observation(inputs="CD", target="GFFG", context={"step_number": 3}),
        ]

    def seed_program_sources(self):
        return []


class TypedPriorBranchAdapter(SeedAdapter):
    def collect_observations(self):
        return [
            Observation(inputs="", target="ACBD", context={"main": "AB", "vice": "CD", "step_number": 1}),
            Observation(inputs="", target="CADB", context={"main": "AB", "vice": "CD", "step_number": 2}),
            Observation(inputs="", target="EAGB", context={"main": "EG", "vice": "AB", "step_number": 3}),
            Observation(inputs="", target="AEBG", context={"main": "EG", "vice": "AB", "step_number": 4}),
        ]

    def seed_program_sources(self):
        return []

    def branch_seed_programs(self) -> bool:
        return True

    def branch_predicate_sources(self, observations):
        return [
            {
                "name": "step_even",
                "expr": "int(context.get('step_number', 0)) % 2 == 0",
            }
        ]


class NumericTypedPriorAdapter(CPIAdapter):
    name = "numeric_prior_test"

    def interface_description(self) -> str:
        return "Toy numeric law task."

    def signature_hint(self) -> str:
        return "def law(inputs: dict) -> float"

    def collect_observations(self):
        return [
            Observation(inputs={"m1": 2.0, "m2": 3.0, "r": 2.0}, target=10.5),
            Observation(inputs={"m1": 4.0, "m2": 5.0, "r": 2.0}, target=35.0),
            Observation(inputs={"m1": 6.0, "m2": 2.0, "r": 3.0}, target=28.0 / 3.0),
            Observation(inputs={"m1": 3.0, "m2": 7.0, "r": 1.0}, target=147.0),
        ]

    def execute(self, program, obs):
        k = getattr(program, "_const", 1.0)
        return k * program.fn(obs.inputs)

    def calibrate(self, program, train_observations):
        ratios = []
        for obs in train_observations:
            base = float(program.fn(obs.inputs))
            if abs(base) > 1e-12:
                ratios.append(float(obs.target) / base)
        if ratios:
            program._const = sum(ratios) / len(ratios)
        return program

    def loss(self, prediction, obs) -> float:
        rel = abs(float(prediction) - float(obs.target)) / (abs(float(obs.target)) + 1e-9)
        return min(1.0, rel)

    def render_report(self, winners, observations) -> str:
        return winners[0][0].description if winners else ""


class NoSeedAppendXAdapter(SeedAdapter):
    def seed_program_sources(self):
        return []


class IdentityOnlyAppendXAdapter(SeedAdapter):
    def seed_program_sources(self):
        return [
            {
                "name": "identity_only",
                "description": "return input unchanged",
                "code": "def rule(current: str, context: dict) -> str:\n    return current\n",
            }
        ]


class GeneratedLayerEngine(UniversalCPI):
    def _propose_self_layers(self, adapter, observations, failure_context=""):
        code = (
            "def build_layer(context: dict) -> dict:\n"
            "    if str(context.get('signature_hint', '')) != "
            "'def rule(current: str, context: dict) -> str':\n"
            "        return {'program_sources': [], 'signals': []}\n"
            "    return {\n"
            "        'program_sources': [{\n"
            "            'name': 'generated_append_x',\n"
            "            'description': 'append x generated by a self-written layer',\n"
            "            'complexity': 1.2,\n"
            "            'code': \"def rule(current: str, context: dict) -> str:\\n    return current + 'x'\\n\",\n"
            "        }],\n"
            "        'signals': ['append constant suffix x'],\n"
            "        'metadata': {'test': 'generated_layer'},\n"
            "    }\n"
        )
        proposal = {
            "name": "generated_append_x_layer",
            "layer_type": "program_source_generator",
            "description": "generate append-x candidate from context",
            "contract": {
                "input": "context: dict",
                "output": "dict(program_sources, signals, residual_hints, metadata)",
            },
            "code": code,
        }
        programs, errors, output = self._execute_layer_source(
            adapter,
            observations,
            layer_name="generated_append_x_layer",
            code=code,
            layer_origin="proposed",
            failure_context=failure_context,
            metadata=proposal,
        )
        return programs, [proposal], errors, [output] if output else []


class GeneratedOperatorLayerEngine(UniversalCPI):
    def _propose_self_layers(self, adapter, observations, failure_context=""):
        program_code = "def rule(current: str, context: dict) -> str:\n    return current + 'x'\n"
        operator_code = (
            "def transform(program_sources: list, context: dict) -> list:\n"
            "    return [{\n"
            "        'name': 'operator_append_x',\n"
            "        'description': 'append x generated by operator',\n"
            "        'complexity': 1.4,\n"
            f"        'code': {program_code!r},\n"
            "    }]\n"
        )
        code = (
            "def build_layer(context: dict) -> dict:\n"
            "    if str(context.get('signature_hint', '')) != "
            "'def rule(current: str, context: dict) -> str':\n"
            "        return {'operator_sources': [], 'signals': []}\n"
            "    return {\n"
            "        'operator_sources': [{\n"
            "            'name': 'append_x_operator',\n"
            "            'description': 'generate a suffix-appending variant from the candidate pool',\n"
            f"            'code': {operator_code!r},\n"
            "        }],\n"
            "        'signals': ['operator creates suffix variants'],\n"
            "        'metadata': {'test': 'generated_operator_layer'},\n"
            "    }\n"
        )
        proposal = {
            "name": "generated_append_x_operator_layer",
            "layer_type": "search_operator_generator",
            "description": "generate an operator that creates append-x variants",
            "contract": {
                "input": "context: dict",
                "output": "dict(operator_sources, signals, residual_hints, metadata)",
            },
            "code": code,
        }
        programs, errors, output = self._execute_layer_source(
            adapter,
            observations,
            layer_name="generated_append_x_operator_layer",
            code=code,
            layer_origin="proposed",
            failure_context=failure_context,
            metadata=proposal,
        )
        operator_programs, operator_errors = self._programs_from_layer_operators(
            adapter,
            [output] if output else [],
            [],
            observations,
            failure_context=failure_context,
        )
        return programs + operator_programs, [proposal], errors + operator_errors, [output] if output else []


class UniversalCPISeedTests(unittest.TestCase):
    def test_layer_runtime_gate_rejects_program_with_undefined_closure(self):
        engine = UniversalCPI(model="unused", max_rounds=0, n_proposals=0)
        code = (
            "def build_layer(context: dict) -> dict:\n"
            "    return {'program_sources': [{\n"
            "        'name': 'bad_closure',\n"
            "        'description': 'invalid hidden closure',\n"
            "        'complexity': 1.0,\n"
            "        'code': \"def rule(current: str, context: dict) -> str:\\n"
            "    return current + observations[0]\\n\",\n"
            "    }]}\n"
        )
        programs, errors, output = engine._execute_layer_source(
            SeedAdapter(),
            SeedAdapter().collect_observations(),
            layer_name="bad_closure_layer",
            code=code,
            layer_origin="proposed",
        )
        self.assertIsNotNone(output)
        self.assertEqual(programs, [])
        self.assertTrue(any("runtime failed" in error for error in errors))

    def test_layer_runtime_gate_keeps_self_contained_program(self):
        engine = UniversalCPI(model="unused", max_rounds=0, n_proposals=0)
        code = (
            "def build_layer(context: dict) -> dict:\n"
            "    return {'program_sources': [{\n"
            "        'name': 'append_x_layer',\n"
            "        'description': 'self-contained program',\n"
            "        'complexity': 1.0,\n"
            "        'code': \"def rule(current: str, context: dict) -> str:\\n"
            "    return current + 'x'\\n\",\n"
            "    }]}\n"
        )
        programs, errors, _ = engine._execute_layer_source(
            SeedAdapter(),
            SeedAdapter().collect_observations(),
            layer_name="append_x_layer",
            code=code,
            layer_origin="proposed",
        )
        self.assertEqual(errors, [])
        self.assertEqual([program.name for program in programs], ["append_x_layer"])

    def test_typed_monomial_plan_closes_a_numeric_residual_class(self):
        adapter = NumericTypedPriorAdapter()
        observations = adapter.collect_observations()
        spec = compile_typed_operator_plan(
            {"family": "positive_monomial_lattice"},
            input_type="dict",
            target_type="float",
            signature_hint=adapter.signature_hint(),
        )
        assert spec is not None
        engine = UniversalCPI(model="unused", max_rounds=0, n_proposals=0)
        programs, errors, output = engine._execute_layer_source(
            adapter,
            observations,
            layer_name="typed_monomial",
            code=spec["code"],
            layer_origin="proposed",
            metadata=spec,
        )
        self.assertIsNotNone(output)
        self.assertEqual(errors, [])
        ranked = engine._rank(programs, observations, adapter)
        self.assertLess(ranked[0][1].loss_mean, 1e-9)

    def test_loaded_operator_receives_adapter_calibration(self):
        """A persisted structural program must fit local constants on reload."""

        adapter = NumericTypedPriorAdapter()
        spec = compile_typed_operator_plan(
            {"family": "positive_monomial_lattice"},
            input_type="dict",
            target_type="float",
            signature_hint=adapter.signature_hint(),
        )
        assert spec is not None
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            SelfLayerRegistry(root).promote(
                namespace="universal",
                name="typed_monomial",
                layer_type="residual_class_operator",
                code=spec["code"],
                contract=spec["contract"],
                description="test typed operator",
                score={"gain": 1.0, "loss_mean": 0.0, "validation_key": "source"},
                min_gain=0.0,
                max_loss_mean=1.0,
            )
            with patch.dict(
                "os.environ",
                {
                    "MARS_SELF_LAYER_ROOT": str(root),
                    "MARS_SELF_LAYER_NAMESPACE": "universal",
                    "MARS_LOAD_TRUSTED_LAYERS": "0",
                    "MARS_LOAD_CANDIDATE_LAYERS": "1",
                    "MARS_CANDIDATE_LAYER_BUDGET": "1",
                },
                clear=False,
            ):
                engine = UniversalCPI(model="unused", max_rounds=0, n_proposals=0)
                programs, errors, _ = engine._load_self_layer_programs(
                    adapter, adapter.collect_observations()
                )
        self.assertEqual(errors, [])
        ranked = engine._rank(programs, adapter.collect_observations(), adapter)
        self.assertLess(ranked[0][1].loss_mean, 1e-9)

    def test_fixed_language_flags_remove_numeric_prior_families(self):
        with patch.dict(
            "os.environ",
            {"MARS_TYPED_PRIORS": "0", "MARS_RESIDUAL_KERNEL": "0", "MARS_DPSR": "0"},
            clear=False,
        ):
            result = UniversalCPI(model="unused", max_rounds=0, n_proposals=0).run(
                NumericTypedPriorAdapter()
            )
        self.assertFalse(
            any("typed_numeric_prior" in program.tags for program, _ in result.winners)
        )

    def test_trusted_seed_can_solve_without_llm_round(self):
        engine = UniversalCPI(model="unused", max_rounds=3)
        result = engine.run(SeedAdapter())
        self.assertEqual(result.rounds, 0)
        self.assertEqual(result.n_proposed, 0)
        self.assertGreaterEqual(result.n_valid, 1)
        self.assertEqual(result.winners[0][0].name, "append_x")
        self.assertEqual(result.winners[0][1].loss_mean, 0.0)

    def test_trusted_seed_composition_can_solve_without_llm_round(self):
        engine = UniversalCPI(model="unused", max_rounds=3)
        result = engine.run(CompositionSeedAdapter())
        self.assertEqual(result.rounds, 0)
        self.assertEqual(result.n_proposed, 0)
        self.assertGreaterEqual(result.n_valid, 4)
        self.assertIn("compose_append_x_then_append_y", result.winners[0][0].name)
        self.assertEqual(result.winners[0][1].loss_mean, 0.0)

    def test_trusted_seed_branching_can_solve_without_llm_round(self):
        engine = UniversalCPI(model="unused", max_rounds=3)
        result = engine.run(BranchCompositionSeedAdapter())
        self.assertEqual(result.rounds, 0)
        self.assertEqual(result.n_proposed, 0)
        self.assertGreaterEqual(result.n_valid, 4)
        self.assertIn("branch_step_even_append_x_else_append_y", result.winners[0][0].name)
        self.assertEqual(result.winners[0][1].loss_mean, 0.0)

    def test_typed_prior_can_solve_without_llm_round(self):
        engine = UniversalCPI(model="unused", max_rounds=3)
        result = engine.run(TypedPriorAdapter())
        self.assertEqual(result.rounds, 0)
        self.assertEqual(result.n_proposed, 0)
        self.assertIn("typed_reverse_shift_concat_shift_step_number", result.winners[0][0].name)
        self.assertEqual(result.winners[0][1].loss_mean, 0.0)

    def test_typed_prior_branching_can_solve_without_llm_round(self):
        engine = UniversalCPI(model="unused", max_rounds=3)
        result = engine.run(TypedPriorBranchAdapter())
        self.assertEqual(result.rounds, 0)
        self.assertEqual(result.n_proposed, 0)
        self.assertIn("branch_step_even_typed_interleave_vice_main_else_typed_interleave_main_vice", result.winners[0][0].name)
        self.assertEqual(result.winners[0][1].loss_mean, 0.0)

    def test_structural_residual_context_is_benchmark_agnostic(self):
        engine = UniversalCPI(model="unused", max_rounds=1)
        adapter = SeedAdapter()
        bad_code = "def rule(current: str, context: dict) -> str:\n    return current\n"
        ok, err, fn = _sandbox_compile(bad_code)
        self.assertTrue(ok, err)
        program = HypothesisProgram(
            name="bad_identity",
            description="identity",
            code=bad_code,
            fn=fn,
        )
        context = engine._structural_residual_context(
            program,
            adapter.collect_observations(),
            adapter,
        )
        self.assertIn("STRUCTURAL INTERFACE RESIDUALS", context)
        self.assertIn("categorical_mismatch", context)

    def test_numeric_typed_prior_can_solve_inverse_square_law_without_llm_round(self):
        engine = UniversalCPI(model="unused", max_rounds=3)
        result = engine.run(NumericTypedPriorAdapter())
        self.assertEqual(result.rounds, 0)
        self.assertEqual(result.n_proposed, 0)
        self.assertIn("typed_numeric_product_m1_m2_over_r_square", result.winners[0][0].name)
        self.assertAlmostEqual(result.winners[0][1].loss_mean, 0.0, places=8)

    def test_numeric_residual_geometry_is_identifier_free_and_log_structured(self):
        adapter = NumericTypedPriorAdapter()
        geometry = UniversalCPI._numeric_residual_geometry(
            adapter,
            adapter.collect_observations(),
            None,
        )
        self.assertEqual(geometry, "positive_log_structured")

    def test_residual_operator_layer_repairs_bad_seed_without_llm_round(self):
        with tempfile.TemporaryDirectory() as td:
            env = {
                "MARS_SELF_LAYER_ROOT": td,
                "MARS_SELF_WRITE_LAYERS": "1",
                "MARS_SELF_WRITE_MODULES": "0",
                "MARS_RESIDUAL_OPERATOR_LAYERS": "1",
            }
            with patch.dict("os.environ", env, clear=False):
                result = UniversalCPI(model="unused", max_rounds=0, n_proposals=0).run(
                    IdentityOnlyAppendXAdapter()
                )
        self.assertEqual(result.rounds, 0)
        self.assertGreaterEqual(result.n_proposed, 1)
        self.assertEqual(result.winners[0][1].loss_mean, 0.0)
        self.assertTrue(
            result.winners[0][0].name.startswith("residual_")
            or "residual_common_suffix" in result.winners[0][0].name
        )
        self.assertTrue(
            any(p.get("kind") == "residual_operator_layer" for p in result.proposals_raw),
            result.proposals_raw,
        )

    def test_self_written_module_becomes_trusted_and_reusable(self):
        with tempfile.TemporaryDirectory() as td:
            base_env = {
                "MARS_SELF_MODULE_ROOT": td,
                "MARS_SELF_WRITE_MODULES": "1",
                "MARS_SELF_MODULE_MIN_KEYS": "2",
            }
            with patch.dict("os.environ", {**base_env, "MARS_VALIDATION_KEY": "task_a"}, clear=False):
                first = UniversalCPI(model="unused", max_rounds=0).run(SeedAdapter())
            self.assertTrue(
                any("promoted self module" in err for err in first.errors),
                first.errors,
            )

            with patch.dict("os.environ", {**base_env, "MARS_VALIDATION_KEY": "task_b"}, clear=False):
                second = UniversalCPI(model="unused", max_rounds=0).run(SeedAdapter())
            self.assertTrue(
                any("trusted library size for seed_test" in err for err in second.errors),
                second.errors,
            )

            with patch.dict("os.environ", base_env, clear=False):
                reused = UniversalCPI(model="unused", max_rounds=0).run(NoSeedAppendXAdapter())
            self.assertEqual(reused.rounds, 0)
            self.assertEqual(reused.n_proposed, 0)
            self.assertEqual(reused.winners[0][1].loss_mean, 0.0)
            self.assertIn("self_module_trusted", reused.winners[0][0].tags)

    def test_self_written_layer_becomes_trusted_and_reusable(self):
        with tempfile.TemporaryDirectory() as td:
            base_env = {
                "MARS_SELF_LAYER_ROOT": td,
                "MARS_SELF_WRITE_LAYERS": "1",
                "MARS_SELF_LAYER_MIN_KEYS": "2",
                "MARS_SELF_WRITE_MODULES": "0",
                "MARS_RESIDUAL_OPERATOR_LAYERS": "0",
            }
            with patch.dict("os.environ", {**base_env, "MARS_VALIDATION_KEY": "layer_task_a"}, clear=False):
                first = UniversalCPI(model="unused", max_rounds=0).run(SeedAdapter())
            self.assertTrue(
                any("promoted self layer" in err for err in first.errors),
                first.errors,
            )

            with patch.dict("os.environ", {**base_env, "MARS_VALIDATION_KEY": "layer_task_b"}, clear=False):
                second = UniversalCPI(model="unused", max_rounds=0).run(SeedAdapter())
            self.assertTrue(
                any("trusted self-layer library size" in err for err in second.errors),
                second.errors,
            )

            with patch.dict("os.environ", base_env, clear=False):
                reused = UniversalCPI(model="unused", max_rounds=0).run(NoSeedAppendXAdapter())
            self.assertEqual(reused.rounds, 0)
            self.assertEqual(reused.n_proposed, 0)
            self.assertEqual(reused.winners[0][1].loss_mean, 0.0)
            self.assertIn("self_layer_program", reused.winners[0][0].tags)

    def test_generated_operator_layer_source_is_promoted_and_reused(self):
        with tempfile.TemporaryDirectory() as td:
            base_env = {
                "MARS_SELF_LAYER_ROOT": td,
                "MARS_SELF_WRITE_LAYERS": "1",
                "MARS_SELF_LAYER_MIN_KEYS": "2",
                "MARS_SELF_WRITE_MODULES": "0",
                "MARS_RESIDUAL_OPERATOR_LAYERS": "0",
            }
            with patch.dict("os.environ", {**base_env, "MARS_VALIDATION_KEY": "op_layer_a"}, clear=False):
                first = GeneratedOperatorLayerEngine(model="unused", max_rounds=1, n_proposals=1).run(NoSeedAppendXAdapter())
            self.assertEqual(first.winners[0][1].loss_mean, 0.0)
            self.assertTrue(
                any("promoted self layer generated_append_x_operator_layer" in err for err in first.errors),
                first.errors,
            )

            with patch.dict("os.environ", {**base_env, "MARS_VALIDATION_KEY": "op_layer_b"}, clear=False):
                second = GeneratedOperatorLayerEngine(model="unused", max_rounds=1, n_proposals=1).run(NoSeedAppendXAdapter())
            self.assertTrue(
                any("trusted self-layer library size" in err for err in second.errors),
                second.errors,
            )

            with patch.dict("os.environ", base_env, clear=False):
                reused = UniversalCPI(model="unused", max_rounds=0, n_proposals=0).run(NoSeedAppendXAdapter())
            self.assertEqual(reused.rounds, 0)
            self.assertEqual(reused.n_proposed, 0)
            self.assertEqual(reused.winners[0][1].loss_mean, 0.0)
            self.assertIn("operator", reused.winners[0][0].tags)

    def test_generated_self_layer_source_is_promoted_and_reused(self):
        with tempfile.TemporaryDirectory() as td:
            base_env = {
                "MARS_SELF_LAYER_ROOT": td,
                "MARS_SELF_WRITE_LAYERS": "1",
                "MARS_SELF_LAYER_MIN_KEYS": "2",
                "MARS_SELF_WRITE_MODULES": "0",
                "MARS_RESIDUAL_OPERATOR_LAYERS": "0",
            }
            with patch.dict("os.environ", {**base_env, "MARS_VALIDATION_KEY": "gen_layer_a"}, clear=False):
                first = GeneratedLayerEngine(model="unused", max_rounds=1, n_proposals=1).run(NoSeedAppendXAdapter())
            self.assertEqual(first.winners[0][1].loss_mean, 0.0)
            self.assertTrue(
                any("promoted self layer generated_append_x_layer" in err for err in first.errors),
                first.errors,
            )

            with patch.dict("os.environ", {**base_env, "MARS_VALIDATION_KEY": "gen_layer_b"}, clear=False):
                second = GeneratedLayerEngine(model="unused", max_rounds=1, n_proposals=1).run(NoSeedAppendXAdapter())
            self.assertTrue(
                any("trusted self-layer library size" in err for err in second.errors),
                second.errors,
            )

            with patch.dict("os.environ", base_env, clear=False):
                reused = UniversalCPI(model="unused", max_rounds=0, n_proposals=0).run(NoSeedAppendXAdapter())
            self.assertEqual(reused.rounds, 0)
            self.assertEqual(reused.n_proposed, 0)
            self.assertEqual(reused.winners[0][1].loss_mean, 0.0)
            self.assertIn("self_layer_program", reused.winners[0][0].tags)


if __name__ == "__main__":
    unittest.main()
