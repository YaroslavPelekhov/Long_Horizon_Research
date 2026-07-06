import numpy as np
import pandas as pd

from mars.skills.contract_baselines import (
    generate_contract_baselines,
    generate_dataframe_analyzer_baselines,
)
from mars.skills.task_contract import compile_task_contract


def test_contract_baselines_generate_auto_npy_mapping(tmp_path):
    train = tmp_path / "benchmark" / "datasets" / "arr" / "train"
    test = tmp_path / "benchmark" / "datasets" / "arr" / "test"
    train.mkdir(parents=True)
    test.mkdir(parents=True)
    np.save(train / "sub01.npy", np.zeros((2, 5, 3)))
    np.save(train / "sub03.npy", np.ones((2, 5, 3)))
    np.save(test / "sub01.npy", np.zeros((2, 4, 3)))
    eval_source = """
from scipy.stats import spearmanr
import numpy as np
def eval():
    arr = np.load("./pred_results/sub01tosub03.npy")
    return int(spearmanr(arr.flatten(), arr.flatten())[0] >= 0.6), ''
"""
    contract = compile_task_contract(
        repo_root=tmp_path,
        dataset_tree="|-- arr/\n|---- train/\n|------ sub01.npy\n|------ sub03.npy\n|---- test/\n|------ sub01.npy",
        output_path="pred_results/sub01tosub03.npy",
        eval_source=eval_source,
        task_text="Map Sub 01 to Sub 03.",
    )

    baselines = generate_contract_baselines(contract, task_text="Map Sub 01 to Sub 03.")

    assert baselines
    assert baselines[0].name == "paired_npy_ridge_mapping"
    assert baselines[0].auto_accept is True
    assert "Ridge" in baselines[0].code
    assert "sub03.npy" in baselines[0].code


def test_contract_baselines_generate_non_auto_csv_scaffold(tmp_path):
    data = tmp_path / "benchmark" / "datasets" / "demo"
    data.mkdir(parents=True)
    pd.DataFrame({"index": ["a"], "target": [0.2], "x": [1.0]}).to_csv(data / "train.csv", index=False)
    pd.DataFrame({"index": ["b"], "target": [-999], "x": [2.0]}).to_csv(data / "test.csv", index=False)
    eval_source = """
from sklearn.metrics import roc_auc_score
import pandas as pd
def eval():
    pred = pd.read_csv("pred_results/pred.csv")
    return int(roc_auc_score([1], pred["target"]) >= 0.5), ''
"""
    contract = compile_task_contract(
        repo_root=tmp_path,
        dataset_tree="|-- demo/\n|---- train.csv\n|---- test.csv",
        output_path="pred_results/pred.csv",
        eval_source=eval_source,
        task_text="Predict target.",
    )

    baselines = generate_contract_baselines(contract, task_text="Predict target.")
    csv = next(b for b in baselines if b.name == "csv_metric_regression_scaffold")

    assert csv.auto_accept is False
    assert "RandomForestRegressor" in csv.code
    assert "target_col = 'target'" in csv.code


def test_dataframe_analyzer_baselines_execute_with_evidence():
    df = pd.DataFrame(
        {
            "year": [2000, 2001, 2002, 2003],
            "education spending": [1.0, 2.0, 3.0, 4.0],
            "gdp": [10.0, 20.0, 30.0, 40.0],
            "noise": [4.0, 1.0, 3.0, 2.0],
        }
    )

    baselines = generate_dataframe_analyzer_baselines(
        df,
        question="How is education spending associated with GDP?",
        column_descriptions={"gdp": "gross domestic product"},
    )

    assert {b.name for b in baselines} >= {
        "df_relevant_correlation_probe",
        "df_temporal_or_axis_extrema_probe",
    }
    ns = {"pd": pd, "np": np}
    exec(baselines[0].code, ns)
    out = ns["analyze"](df)

    assert out["statistic"] > 0
    assert "education" in out["evidence"].lower() or "gdp" in out["evidence"].lower()
