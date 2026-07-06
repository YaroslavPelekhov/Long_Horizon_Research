import pandas as pd

from mars.induction.universal_hypothesis_kernel import (
    KernelCandidate,
    KernelTask,
    TypedHole,
    UniversalHypothesisKernel,
    _invalid_evidence_penalty,
)


def test_kernel_closes_temporal_slots_with_executable_operator():
    df = pd.DataFrame(
        {
            "BCE": [4000, 3500, 3200, 2800],
            "AxesCelts_inter": [0.1, 0.5, 2.6, 0.2],
            "Other": [1.0, 2.0, 3.0, 4.0],
        }
    )

    result = UniversalHypothesisKernel().run(
        KernelTask(
            task_text="In which century did the axes become quantitatively most frequent?",
            interface_kind="discovery_table",
            data=df,
            schema={"BCE": "time in BCE", "AxesCelts_inter": "axes and celts"},
        )
    )

    assert result.best is not None
    assert result.best.coverage >= 0.75
    assert "UniversalHypothesisKernel" in result.best.report()
    assert "AxesCelts" in result.best.report()


def test_kernel_posterior_prefers_complete_low_complexity_candidate():
    class FakeOperator:
        name = "fake"
        complexity = 1.0

        def propose(self, task, contract):
            _ = task, contract
            partial = KernelCandidate(
                hypothesis="partial",
                workflow="missed slots",
                evidence="evidence",
                answer_form="temporal_event",
                holes=(
                    TypedHole("event", value="peak"),
                    TypedHole("variable", value=None),
                    TypedHole("time", value=None),
                    TypedHole("relation", value="maximum"),
                ),
                operator="complex",
                source="table",
                evidence_score=20.0,
                complexity=5.0,
                coverage=0.5,
                loss=0.5,
                posterior=16.0,
            )
            complete = KernelCandidate(
                hypothesis="complete",
                workflow="closed all typed holes",
                evidence="evidence",
                answer_form="temporal_event",
                holes=(
                    TypedHole("event", value="peak"),
                    TypedHole("variable", value="axes"),
                    TypedHole("time", value="3200 BCE"),
                    TypedHole("relation", value="maximum"),
                ),
                operator="simple",
                source="table",
                evidence_score=8.0,
                complexity=1.0,
                coverage=1.0,
                loss=0.0,
                posterior=33.0,
            )
            return [partial, complete]

    result = UniversalHypothesisKernel(operators=[FakeOperator()]).run(
        KernelTask(
            task_text="In which century did the axes become quantitatively most frequent?",
            interface_kind="discovery_table",
        )
    )

    assert result.best is not None
    assert result.best.hypothesis == "complete"


def test_kernel_penalizes_empty_evidence_reports():
    bad = (
        "HYPOTHESIS: the available evidence does not support a simple direct effect. "
        "Evidence: Not enough relevant numeric columns for association probe."
    )
    good = "HYPOTHESIS: Urban land use and elevation have a measured interaction coefficient of 0.22."

    assert _invalid_evidence_penalty(bad) > 30
    assert _invalid_evidence_penalty(good) == 0
