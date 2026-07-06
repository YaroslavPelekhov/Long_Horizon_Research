import numpy as np
import pandas as pd

from mars.induction.estimand_synthesizer import infer_estimand_hypothesis
from mars.induction.role_canonicalizer import select_interaction_roles
from mars.induction.universal_hypothesis_kernel import KernelTask, UniversalHypothesisKernel


def test_estimand_synthesizer_builds_latent_ses_instead_of_id():
    rng = np.random.default_rng(11)
    n = 240
    income = rng.normal(size=n)
    mother = income + rng.normal(scale=0.5, size=n)
    father = income + rng.normal(scale=0.5, size=n)
    logits = -0.8 + 1.5 * income
    y = rng.random(n) < 1 / (1 + np.exp(-logits))
    df = pd.DataFrame(
        {
            "ID": np.arange(n),
            "Total net family income, previous calendar year, 1979": income,
            "Highest grade completed by respondent's mother, 1979": mother,
            "Highest grade completed by respondent's father, 1979": father,
            "Highest grade completed, 1979": np.where(y, 16, 12),
        }
    )

    result = infer_estimand_hypothesis(
        question="How does socioeconomic status affect the likelihood of completing a BA degree?",
        data=df,
        schema={},
    )

    assert result is not None
    assert "__latent_socioeconomic_status" in result.evidence
    assert "ID" not in result.evidence
    assert "positive relationship" in result.hypothesis


def test_survey_estimand_does_not_count_not_quoted_as_quoted():
    df = pd.DataFrame(
        {
            "Q8_ML_Addressing_Business_Analyst": ["Quoted"] * 3 + ["Not quoted"] * 7,
            "Q8_ML_Addressing_Developer": ["Quoted"] * 2 + ["Not quoted"] * 8,
        }
    )

    result = infer_estimand_hypothesis(
        question=(
            "What are the proportions and their 95% Confidence Intervals for "
            "Business Analysts and Developers who are associated with addressing requirements?"
        ),
        data=df,
        schema={},
    )

    assert result is not None
    assert "Business Analyst" in result.hypothesis
    assert "Developer" in result.hypothesis
    assert "30.000%" in result.hypothesis
    assert "20.000%" in result.hypothesis


def test_interaction_estimand_uses_context_to_choose_factors():
    rng = np.random.default_rng(4)
    n = 180
    urban = rng.normal(size=n)
    elevation = rng.normal(size=n)
    total = np.full(n, 10.0)
    prop = 0.4 + 0.2 * urban * elevation + rng.normal(scale=0.03, size=n)
    gard = np.clip(prop * total, 0, total)
    df = pd.DataFrame(
        {
            "n.gard": gard,
            "n.unint": total - gard,
            "n.total": total,
            "urban.2009.50m": urban,
            "elevation": elevation,
        }
    )

    result = infer_estimand_hypothesis(
        question="What factors interact significantly to affect the proportion of gardening-introduced plants?",
        domain_context="Sibling question: How do urban land use and elevation interact?",
        data=df,
        schema={},
    )

    assert result is not None
    assert "estimand_role_interaction:urban_land_use" in result.evidence
    assert "*elevation" in result.evidence
    assert "urban land use" in result.hypothesis
    assert result.slots["operator"] == "interaction"
    assert result.slots["target"] == "the proportion of gardening-introduced non-native plants"
    assert "measured_value" in result.slots
    assert "n.gard*n.unint" not in result.evidence


def test_role_canonicalizer_compresses_proxy_columns_before_interaction():
    rng = np.random.default_rng(7)
    n = 120
    urban = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "n.gard": np.ones(n),
            "n.total": np.full(n, 5),
            "urban.1956.50m": urban + rng.normal(scale=0.1, size=n),
            "urban.1993.50m": urban + rng.normal(scale=0.1, size=n),
            "urban.2009.50m": urban + rng.normal(scale=0.1, size=n),
            "elevation": rng.normal(size=n),
        }
    )

    roles = select_interaction_roles(
        task_text="How do urban land use and elevation interact?",
        domain_context="",
        df=df,
        schema={},
        target="n.gard",
    )

    assert len(roles) >= 2
    assert roles[0].name == "urban_land_use"
    assert roles[0].column == "__role_urban_land_use"
    assert set(roles[0].source_columns) >= {"urban.1956.50m", "urban.1993.50m", "urban.2009.50m"}
    assert roles[1].name == "elevation"


def test_role_outcome_contrast_compares_primary_outcome_types():
    rng = np.random.default_rng(13)
    n = 160
    urban = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "urban.1956.50m": urban + rng.normal(scale=0.1, size=n),
            "urban.1993.50m": urban + rng.normal(scale=0.1, size=n),
            "n.gard": 1.0 + 0.25 * urban + rng.normal(scale=0.2, size=n),
            "n.unint": 1.5 + 0.75 * urban + rng.normal(scale=0.2, size=n),
            "n.total": 3.0 + urban,
            "elevation": 100 + 50 * urban,
            "distance.stream": 20 - urban,
        }
    )

    result = infer_estimand_hypothesis(
        question="How did urban land use affect the invasion of different types of introduced plants?",
        data=df,
        schema={},
    )

    assert result is not None
    assert result.slots["estimand_family"] == "role_conditioned_outcome_contrast"
    assert "urban land use reduced invasion by gardening plants over unintentionally introduced ones" in result.hypothesis.lower()
    assert "estimand_role_outcome_contrast" in result.evidence


def test_kernel_prefers_estimand_criminal_history_over_race_proxy():
    df = pd.DataFrame(
        {
            "Racial/ethnic cohort, 1979": [1, 2, 3, 1, 2, 3],
            "Ever convicted of an illegal act in adult court before 1980": [1, 1, 0, 0, 0, 0],
            "Family net wealth, 1996 (key data point)": [5, 6, 100, 90, 80, 70],
        }
    )

    result = UniversalHypothesisKernel().run(
        KernelTask(
            task_text="How does having a criminal history influence wealth levels compared to those without such a history?",
            interface_kind="discovery_table",
            data=df,
            schema={},
        )
    )

    assert result.best is not None
    assert result.best.operator == "estimand_synthesizer"
    assert "criminal history" in result.best.hypothesis
