import numpy as np
import pandas as pd

from mars.skills.task_contract import compile_task_contract


def test_task_contract_finds_csv_target_placeholder_and_features(tmp_path):
    data = tmp_path / "benchmark" / "datasets" / "demo"
    data.mkdir(parents=True)
    pd.DataFrame(
        {
            "index": ["a", "b"],
            "Signal-inhibition": [0.2, 0.8],
            "feat_a": [1, 2],
            "feat_b": [3, 4],
        }
    ).to_csv(data / "demo_train.csv", index=False)
    pd.DataFrame(
        {
            "index": ["c"],
            "Signal-inhibition": [-999],
            "feat_a": [5],
            "feat_b": [6],
        }
    ).to_csv(data / "demo_test.csv", index=False)
    eval_source = """
import pandas as pd
def eval():
    pred = pd.read_csv('pred_results/demo_pred.csv')
    gold = pd.read_csv('benchmark/eval_programs/gold_results/demo_gold.csv')
    return int(list(pred["index"]) == list(gold["index"]) and pred["Signal-inhibition"].mean() > 0), ''
"""

    contract = compile_task_contract(
        repo_root=tmp_path,
        dataset_tree="|-- demo/\n|---- demo_train.csv\n|---- demo_test.csv",
        output_path="pred_results/demo_pred.csv",
        eval_source=eval_source,
        task_text="Predict signal inhibition.",
    )

    assert contract.required_output_columns == ("index", "Signal-inhibition")
    assert contract.target_candidates == ("Signal-inhibition",)
    assert contract.id_columns == ("index",)
    assert set(contract.recommended_features) == {"feat_a", "feat_b"}
    assert "placeholder_columns_detected_do_not_use_as_observed_test_labels" in contract.warnings


def test_task_contract_finds_npy_shapes_and_eval_reshape(tmp_path):
    data = tmp_path / "benchmark" / "datasets" / "arr" / "train"
    test = tmp_path / "benchmark" / "datasets" / "arr" / "test"
    data.mkdir(parents=True)
    test.mkdir(parents=True)
    np.save(data / "sub01.npy", np.zeros((2, 5, 3)))
    np.save(test / "sub01.npy", np.zeros((2, 4, 3)))
    eval_source = """
import numpy as np
def eval():
    arr = np.load("./pred_results/out.npy")
    ref = np.reshape(arr, [4, 6])
    return int(ref.size > 0), ''
"""

    contract = compile_task_contract(
        repo_root=tmp_path,
        dataset_tree="|-- arr/\n|---- train/\n|------ sub01.npy\n|---- test/\n|------ sub01.npy",
        output_path="pred_results/out.npy",
        eval_source=eval_source,
        task_text="Generate output array.",
    )

    assert contract.output_kind == "npy"
    assert any("shape=(2, 5, 3)" in rule for rule in contract.array_shape_rules)
    assert any("4x6" in rule for rule in contract.array_shape_rules)
    assert contract.train_test_pairs
