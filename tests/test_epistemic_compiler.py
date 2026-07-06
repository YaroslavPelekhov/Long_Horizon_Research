from mars.induction.epistemic_compiler import (
    benchmark_leakage_penalty,
    certify_epistemic_result,
    fingerprint_task,
)
from mars.induction.universal_cpi import HypothesisProgram, Observation, ProgramScore


def test_task_fingerprint_scrubs_benchmark_identity():
    observations = [
        Observation(inputs={"x": 1.0}, target=2.0, context={"split": "a"}),
        Observation(inputs={"x": 2.0}, target=4.0, context={"split": "a"}),
        Observation(inputs={"x": 3.0}, target=6.0, context={"split": "b"}),
    ]

    left = fingerprint_task(
        signature_hint="def law(inputs: dict) -> float",
        interface_description="NewtonBench hidden scalar law.",
        observations=observations,
    )
    right = fingerprint_task(
        signature_hint="def law(inputs: dict) -> float",
        interface_description="DiscoveryBench hidden scalar law.",
        observations=observations,
    )

    assert left.interface_family == "numeric_law"
    assert left.observable_hash == right.observable_hash


def test_benchmark_leakage_penalty_detects_local_patch_language():
    generic = "def law(inputs):\n    return inputs['x'] * 2\n"
    local = "def law(inputs):\n    # newtonbench shortcut\n    return inputs['x'] * 2\n"

    assert benchmark_leakage_penalty(generic, adapter_name="newtonbench") == 0.0
    assert benchmark_leakage_penalty(local, adapter_name="newtonbench") > 0.2


def test_epistemic_certificate_accepts_compressive_generic_program():
    observations = [
        Observation(inputs={"x": 1.0}, target=2.0, context={}),
        Observation(inputs={"x": 2.0}, target=4.0, context={}),
        Observation(inputs={"x": 3.0}, target=6.0, context={}),
        Observation(inputs={"x": 4.0}, target=8.0, context={}),
    ]
    program = HypothesisProgram(
        name="generic_linear_measurement",
        description="measure a one-dimensional linear relation",
        code="def law(inputs: dict) -> float:\n    return inputs['x'] * 2\n",
        fn=lambda inputs: inputs["x"] * 2,
        complexity=1.0,
        tags=("measurement_probe",),
    )
    weak = HypothesisProgram(
        name="weak_constant",
        description="constant baseline",
        code="def law(inputs: dict) -> float:\n    return 0.0\n",
        fn=lambda inputs: 0.0,
        complexity=0.5,
    )
    winners = [
        (program, ProgramScore("generic_linear_measurement", 0.0, 1.0, 0.01, 4, 1.0)),
        (weak, ProgramScore("weak_constant", 1.0, 0.0, 1.0, 4, 0.5)),
    ]

    cert = certify_epistemic_result(
        adapter_name="newtonbench",
        signature_hint="def law(inputs: dict) -> float",
        interface_description="Hidden scalar law.",
        observations=observations,
        winners=winners,
        programs=[program, weak],
        proposals_raw=[],
    )

    assert cert.accepted
    assert cert.compression_gain > 10
    assert cert.leakage_penalty == 0.0


def test_epistemic_certificate_rejects_benchmark_named_patch():
    observations = [
        Observation(inputs={"x": 1.0}, target=2.0, context={}),
        Observation(inputs={"x": 2.0}, target=4.0, context={}),
        Observation(inputs={"x": 3.0}, target=6.0, context={}),
    ]
    program = HypothesisProgram(
        name="newtonbench_patch",
        description="NewtonBench special case",
        code="def law(inputs: dict) -> float:\n    # discoverybench/newtonbench patch\n    return inputs['x'] * 2\n",
        fn=lambda inputs: inputs["x"] * 2,
        complexity=1.0,
    )
    winners = [
        (program, ProgramScore("newtonbench_patch", 0.0, 1.0, 0.01, 3, 1.0)),
    ]

    cert = certify_epistemic_result(
        adapter_name="newtonbench",
        signature_hint="def law(inputs: dict) -> float",
        interface_description="Hidden scalar law.",
        observations=observations,
        winners=winners,
        programs=[program],
        proposals_raw=[],
    )

    assert not cert.accepted
    assert cert.leakage_penalty > 0.2
    assert "winner text/code contains benchmark-specific tokens" in cert.rejected_reasons
