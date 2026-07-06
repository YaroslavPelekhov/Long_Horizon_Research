from mars.runners.run_universal_discovery_real_eval import (
    _report_relevance,
    _should_use_sibling_query_context,
    _satisfies_surface_contract,
    _task_domain_context,
)


def test_relevance_prefers_stated_coefficient_contract_over_generic_category():
    question = "What are the variables between which a positive relationship is quantified by a coefficient of 0.22?"
    coefficient_report = (
        "HYPOTHESIS: There is a relationship between invaded.niche.breadth and native.niche.breadth. "
        "The relation is positive with a coefficient of 0.24.\n"
        "WORKFLOW SUMMARY: Closed slots: answer_form=stated_coefficient_relationship. "
        "Evidence: answer_slot_stated_coefficient:invaded.niche.breadth:native.niche.breadth:coef=0.244:target=0.22."
    )
    generic_report = (
        "HYPOTHESIS: In the selected records, studies primarily used AgriForest for pathway (33.3%).\n"
        "WORKFLOW SUMMARY: Closed slots: answer_form=generic_categorical_measurement. "
        "Evidence: answer_slot_generic_category:pathway:AgriForest:pct=33.3."
    )

    assert _satisfies_surface_contract(question, coefficient_report)
    assert _report_relevance(question, "invasion_success_pathways.csv", coefficient_report) > _report_relevance(
        question,
        "temporal_trends_contingency_table.csv",
        generic_report,
    )


def test_relevance_vetoes_pollen_source_for_non_pollen_artifact_question():
    question = "In which century did the axes become quantitatively most frequent?"
    pollen_report = (
        "HYPOTHESIS: At the middle of the 1st millennium BCE, axes become quantitatively most frequent.\n"
        "WORKFLOW SUMMARY: Evidence: slot_contract_peak:Belau_PC1:time=-704:value=8.714:posterior=1.88."
    )
    artifact_report = (
        "HYPOTHESIS: At the end of the 4th millennium BCE, axes become quantitatively most frequent.\n"
        "WORKFLOW SUMMARY: Evidence: slot_contract_peak:AxesCelts_inter:time=-3200:value=2.6:posterior=29.4."
    )

    noisy_context = question + "\nDomain context: pollen openness landscape variables are present in another file."
    assert _report_relevance(noisy_context, "time_series_data.csv", artifact_report) > _report_relevance(
        noisy_context,
        "pollen_openness_score_Belau_Woserin_Feeser_et_al_2019.csv",
        pollen_report,
    )


def test_task_domain_context_includes_sibling_queries_without_gold_answers():
    context = _task_domain_context(
        "",
        {
            "queries": [
                [
                    {"question": "What are the variables linked by coefficient 0.22?"},
                    {"question": "What is the relationship between urban land use and gardening-introduced plants?"},
                ]
            ]
        },
    )

    assert "Sibling task questions" in context
    assert "urban land use" in context
    assert "coefficient 0.22" in context


def test_sibling_query_context_is_gated_to_underspecified_relation_tasks():
    assert _should_use_sibling_query_context(
        "What are the variables between which a positive relationship is quantified by a coefficient of 0.22?"
    )
    assert not _should_use_sibling_query_context(
        "In which century did the axes become quantitatively most frequent?"
    )
    assert not _should_use_sibling_query_context(
        "In the PCA analysis, what distinguishes the time slice around 3500 BCE?"
    )
