from types import SimpleNamespace

from mars.induction.metamorphic_estimand_kernel import evaluate_metamorphic_estimand
from mars.induction.universal_hypothesis_kernel import KernelTask, UniversalHypothesisKernel
from mars.skills.evidence_contract_compiler import compile_evidence_contract
from mars.skills.answer_contract import normalize_answer_report


def _candidate(*, hypothesis, workflow, evidence, answer_form, holes=()):
    return SimpleNamespace(
        hypothesis=hypothesis,
        workflow=workflow,
        evidence=evidence,
        answer_form=answer_form,
        source="table",
        holes=holes,
    )


def test_mek_penalizes_row_level_numeric_axis_for_period_crossover():
    question = "Over which time period has gardening surpassed agriculture as the main contributor to the non-native flora?"
    contract = compile_evidence_contract(question, interface_kind="discovery_table")
    bad = _candidate(
        hypothesis="Gardening surpassed agriculture from 6 through 250.",
        workflow="Closed slots: answer_form=period_category_crossover, period=mrt.",
        evidence="answer_slot_period_crossover:6;19;23;28;31;36;48;50;51;80;86;95;100;104",
        answer_form="period_crossover",
    )

    evaluation = evaluate_metamorphic_estimand(
        task_text=question,
        domain_context="",
        schema={},
        contract=contract,
        candidate=bad,
    )

    assert evaluation.loss > 0.3
    assert "period_axis_not_categorical" in evaluation.signatures


def test_mek_accepts_categorical_temporal_crossover_axis():
    question = "Over which time period has gardening surpassed agriculture as the main contributor to the non-native flora?"
    contract = compile_evidence_contract(question, interface_kind="discovery_table")
    good = _candidate(
        hypothesis="Over the past millennium, gardening has replaced agriculture.",
        workflow="Closed slots: answer_form=period_category_crossover, period=introduction.period.",
        evidence="answer_slot_period_crossover:1501-1900;1901-1984;1985-2019",
        answer_form="period_crossover",
    )

    evaluation = evaluate_metamorphic_estimand(
        task_text=question,
        domain_context="",
        schema={},
        contract=contract,
        candidate=good,
    )

    assert evaluation.loss == 0.0


def test_mek_penalizes_underspecified_coefficient_without_context_anchor():
    question = "What are the variables between which a positive relationship is quantified by a coefficient of 0.22?"
    contract = compile_evidence_contract(question, interface_kind="discovery_table")
    context = "Sibling task questions: relationship between urban land use and gardening-introduced non-native plants."
    schema = {
        "urban.2009.50m": "degree of urban land use",
        "n.gard": "number of gardening-introduced non-native plants",
        "native.niche.breadth": "native niche breadth",
        "invaded.niche.breadth": "invaded niche breadth",
    }
    wrong = _candidate(
        hypothesis="There is a relationship between invaded niche breadth and native niche breadth.",
        workflow="Closed slots: variables=invaded.niche.breadth and native.niche.breadth.",
        evidence="answer_slot_stated_coefficient:invaded.niche.breadth:native.niche.breadth:coef=0.244:target=0.22",
        answer_form="stated_coefficient_relationship",
    )

    evaluation = evaluate_metamorphic_estimand(
        task_text=question,
        domain_context=context,
        schema=schema,
        contract=contract,
        candidate=wrong,
    )

    assert evaluation.loss > 0.3
    assert "semantic_role_anchor_missing" in evaluation.signatures


def test_normalizer_removes_unasked_population_filter_for_prevalence_question():
    question = "How does the prevalence of non-native plants introduced via gardening vary based on habitat type?"
    report = (
        "HYPOTHESIS: In groups where gardening-introduced non-native plants are present, "
        "the prevalence of gardening-introduced non-native plants differs between urban/cropland "
        "habitats and natural habitats."
    )

    normalized = normalize_answer_report(question, report)

    assert normalized.startswith("The prevalence of gardening-introduced")


def test_kernel_report_contains_metamorphic_signature_for_bad_candidate():
    class BadCrossoverOperator:
        name = "bad_crossover"
        complexity = 1.0

        def propose(self, task, contract):
            from mars.induction.universal_hypothesis_kernel import _candidate_from_parts

            return [
                _candidate_from_parts(
                    contract=contract,
                    hypothesis="Gardening surpassed agriculture from 6 through 250.",
                    workflow="Closed slots: answer_form=period_category_crossover, period=mrt.",
                    evidence="answer_slot_period_crossover:6;19;23;28;31;36;48;50;51;80;86;95;100;104",
                    slots={"answer_form": "period_crossover", "period": "mrt", "category": "pathway", "count": "n"},
                    answer_form="period_crossover",
                    operator=self.name,
                    source="table",
                    evidence_score=30.0,
                    complexity=self.complexity,
                )
            ]

    result = UniversalHypothesisKernel(operators=[BadCrossoverOperator()]).run(
        KernelTask(
            task_text="Over which time period has gardening surpassed agriculture as the main contributor to the non-native flora?",
            interface_kind="discovery_table",
        )
    )

    assert result.best is not None
    assert "period_axis_not_categorical" in result.best.metamorphic_signatures
