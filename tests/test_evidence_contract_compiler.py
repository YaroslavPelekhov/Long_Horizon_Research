import pandas as pd

from mars.skills.evidence_contract_compiler import (
    compile_evidence_contract,
    infer_evidence_contract_hypothesis,
    score_evidence_coverage,
)


def test_temporal_question_compiles_required_slots():
    contract = compile_evidence_contract("In which century did the axes become quantitatively most frequent?")

    assert contract.answer_form == "temporal_event"
    assert contract.required_slot_names == ("event", "variable", "time", "relation")


def test_coefficient_question_compiles_question_form_slots():
    contract = compile_evidence_contract(
        "What are the variables between which a positive relationship is quantified by a coefficient of 0.22?"
    )

    assert contract.answer_form == "stated_coefficient_relationship"
    assert contract.required_slot_names == ("x", "y", "coefficient", "direction")


def test_affect_question_compiles_measured_relation_slots():
    contract = compile_evidence_contract("How did urban land use affect the invasion of different introduced types?")

    assert contract.answer_form == "measured_relation"
    assert contract.required_slot_names == ("variables", "relation", "statistic")

    promote_contract = compile_evidence_contract("In what scenario did urban land use promote a specific type of invasion?")
    assert promote_contract.answer_form == "measured_relation"


def test_coverage_accepts_closed_temporal_report():
    contract = compile_evidence_contract("In which century did the axes become quantitatively most frequent?")
    report = (
        "Closed slots: event=peak, variable=ZBeil, time=3200 BCE, relation=maximum. "
        "Evidence: slot_contract_peak:ZBeil:time=3200:value=2.6."
    )

    coverage = score_evidence_coverage(contract, report, {"event": "peak", "variable": "ZBeil", "time": "3200 BCE"})

    assert coverage.accepted
    assert coverage.missing_slots == ()


def test_infer_evidence_contract_uses_executable_table_probe():
    df = pd.DataFrame(
        {
            "BCE": [4000, 3500, 3200, 2800],
            "ZBeil": [0.1, 0.5, 2.6, 0.2],
            "Other": [9.0, 1.0, 0.0, 3.0],
        }
    )

    result = infer_evidence_contract_hypothesis(
        question="In which century did the axes become quantitatively most frequent?",
        domain_context="",
        data=df,
        schema={"BCE": "time in BCE", "ZBeil": "axes"},
        interface_kind="discovery_table",
    )

    assert result is not None
    assert result.coverage.score >= 0.75
    assert "EvidenceContractCompiler" in result.as_report()
    assert "ZBeil" in result.as_report()
