from unittest.mock import patch

import pandas as pd

from mars.induction.cpi_adapters import DiscoveryAdapter
from mars.induction.universal_cpi import HypothesisProgram
from mars.skills.answer_contract import compile_answer_contract, normalize_answer_report


def test_answer_contract_strips_workflow_and_generic_prefix():
    question = "In which century did the axes become quantitatively most frequent?"
    report = (
        "HYPOTHESIS: In humanities, at the end of the 4th millennium BCE, "
        "axes become quantitatively most frequent.\n"
        "WORKFLOW SUMMARY: debug text that should not be judged."
    )

    contract = compile_answer_contract(question)
    normalized = normalize_answer_report(question, report)

    assert contract.answer_type == "temporal_century"
    assert normalized == (
        "At the end of the 4th millennium BCE, axes become quantitatively most frequent."
    )
    assert "WORKFLOW" not in normalized


def test_answer_contract_renders_minimal_query_relation_when_overrendered():
    question = "How does increased education expenditure influence human capital and economic output?"
    report = (
        "HYPOTHESIS: Adjusted savings: education expenditure (percentage of GNI) "
        "is associated with GNI per capita (constant 2015 USdollar), with Labor "
        "force participation rate, total (percentage of total population ages 15+) "
        "(modeled ILO estimate) acting as an intermediate mechanism or proxy in "
        "the observed data."
    )

    assert normalize_answer_report(question, report) == (
        "Increased education expenditure positively influences human capital and economic output."
    )


def test_answer_contract_uses_question_surface_for_temporal_slot():
    question = "In which century did the axes become quantitatively most frequent?"
    report = (
        "HYPOTHESIS: At the end of the 4th millennium BCE, axes and celts "
        "became quantitatively most frequent."
    )

    assert normalize_answer_report(question, report) == (
        "At the end of the 4th millennium BCE, axes become quantitatively most frequent."
    )


def test_answer_contract_preserves_temporal_onset_clause_from_question():
    question = "In which century did the number of daggers began to increase in importance for the first time?"
    report = "HYPOTHESIS: Daggers began to increase in importance for the first time around 2300/2200 BCE."

    assert normalize_answer_report(question, report) == (
        "Around 2300/2200 BCE, the number of daggers began to increase in importance for the first time."
    )


def test_answer_contract_renders_peak_clause_in_past_tense():
    question = "In which century did the Depots peak?"
    report = "HYPOTHESIS: Around 1100 BCE, Depots peaked."

    assert normalize_answer_report(question, report) == "Around 1100 BCE, the Depots peaked."


def test_answer_contract_does_not_clip_decimal_percentages():
    question = "What type of subjects were used in all replication studies in Experimental Economics?"
    report = (
        "In Experimental Economics, most original studies used student subjects "
        "(94.4%), while all replication studies used student subjects (100.0%)."
    )

    normalized = normalize_answer_report(question, report)

    assert "94.4%" in normalized
    assert "replication studies" in normalized


def test_answer_contract_does_not_clip_dotted_column_names():
    question = "What are the variables between which a positive relationship is quantified by a coefficient of 0.22?"
    report = (
        "HYPOTHESIS: There is a relationship between invaded.niche.breadth and "
        "native.niche.breadth. The relation is positive with a coefficient of 0.24."
    )

    normalized = normalize_answer_report(question, report)

    assert "invaded.niche.breadth" in normalized
    assert "native.niche.breadth" in normalized
    assert "coefficient of 0.24" in normalized


def test_answer_contract_preserves_full_activity_hypothesis_for_hms():
    report = (
        "HYPOTHESIS: Gardening surpassed agriculture as the main contributor to the non-native flora "
        "from 1501-1900 through 1985-2019.\n"
        "WORKFLOW SUMMARY: Evidence: answer_slot_period_crossover:1501-1900;1901-1984;1985-2019."
    )

    normalized = normalize_answer_report(
        "What activity has replaced agriculture as the main contributor to the non-native flora over the past millennium?",
        report,
    )

    assert "Gardening surpassed agriculture" in normalized
    assert "main contributor" in normalized


def test_answer_contract_preserves_full_period_hypothesis_for_hms():
    report = (
        "HYPOTHESIS: Over the past millennium, gardening has replaced agriculture.\n"
        "WORKFLOW SUMMARY: Evidence: answer_slot_period_crossover:1501-1900;1901-1984;1985-2019."
    )

    normalized = normalize_answer_report(
        "Over which time period has gardening surpassed agriculture as the main contributor to the non-native flora?",
        report,
    )

    assert normalized == "Over the past millennium, gardening has replaced agriculture."


def test_answer_contract_preserves_full_variable_pair_hypothesis_for_hms():
    report = (
        "HYPOTHESIS: There is a positive coefficient relationship.\n"
        "WORKFLOW SUMMARY: Closed slots: variables=n.gard and urban.2009.50m, target_coefficient=0.22. "
        "Evidence: answer_slot_stated_coefficient:n.gard:urban.2009.50m:coef=0.231:target=0.22."
    )

    normalized = normalize_answer_report(
        "What are the variables between which a positive relationship is quantified by a coefficient of 0.22?",
        report,
    )

    assert normalized == "There is a positive coefficient relationship."


def test_answer_contract_preserves_multi_sentence_pca_contrast():
    question = (
        "In the PCA analysis of forms of capital during the Early Neolithic period "
        "(4000-3500 BCE), what distinguishes the time slice around 3500 BCE from the general trend?"
    )
    report = (
        "During the Early Neolithic (4000-3500 BCE), the time slices are primarily "
        "characterized by positive values on the first principal component (PC1). "
        "However, the time slice around 3500 BCE is an outlier with a negative value on PC1. "
        "This Principal component analysis (PCA) is on the forms of capital."
    )

    normalized = normalize_answer_report(question, report)

    assert "outlier" in normalized
    assert "negative value on PC1" in normalized
    assert "Principal component analysis" in normalized


def test_discovery_temporal_question_does_not_render_generic_semantic_chain():
    df = pd.DataFrame({"century": ["4th millennium BCE", "3rd millennium BCE"], "axes": [10, 2]})
    adapter = DiscoveryAdapter(
        df,
        "In which century did the axes become quantitatively most frequent?",
        "",
        {"century": "time period", "axes": "artifact frequency"},
        enabled_modules={"temporal"},
    )
    generic_chain = HypothesisProgram(
        name="generic_chain",
        description="generic mediated association",
        code="",
        fn=lambda _df: {
            "cause": "ZMW",
            "mediator": "ZBeil",
            "outcome": "ZCU_AU",
            "relation": "mediated association",
            "evidence": "ZMW, ZBeil, ZCU_AU form a stable chain.",
            "statistic": 1.0,
        },
    )

    with patch(
        "mars.induction.cpi_adapters.infer_temporal_event_hypothesis",
        return_value=(
            "At the end of the 4th millennium BCE, axes become quantitatively most frequent.",
            "Measured artifact frequency by period.",
            "The maximum axes count is in the 4th millennium BCE.",
        ),
    ):
        report = adapter.render_report([(generic_chain, object())], adapter.collect_observations())

    assert "4th millennium BCE" in report
    assert "ZMW" not in report
