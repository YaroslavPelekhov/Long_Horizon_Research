from pathlib import Path

from mars.runners.run_sab_official_eval import _static_program_checks
from mars.skills.code_repair_compiler import compile_code_repairs


def test_static_program_checks_find_missing_dataset_ref_and_unsafe_mapping(tmp_path: Path):
    benchmark = tmp_path / "benchmark"
    (benchmark / "datasets" / "Present").mkdir(parents=True)
    (benchmark / "datasets" / "Present" / "data.csv").write_text("x\n1\n")

    pred_dir = tmp_path / "preds"
    pred_dir.mkdir()
    pred = pred_dir / "pred_example.py"
    pred.write_text(
        "\n".join(
            [
                "import numpy as np",
                "ok = 'benchmark/datasets/Present/data.csv'",
                "bad = 'benchmark/datasets/Missing/file.tif'",
                "mapped = np.vectorize(classification_dict.get)(data)",
            ]
        )
    )

    checks = _static_program_checks(pred_files=[pred], benchmark_path=benchmark)

    assert checks["n_missing_dataset_refs"] == 1
    assert checks["missing_dataset_refs"][0]["program"] == "pred_example.py"
    assert checks["n_unsafe_mapping_get"] == 1


def test_code_repair_compiler_totalizes_numpy_mapping_and_output_dir():
    src = "\n".join(
        [
            "import numpy as np",
            "mapping = {1: 10}",
            "data = np.array([1, 2])",
            "mapped = np.vectorize(mapping.get)(data)",
            "np.save('pred_results/out.npy', mapped)",
        ]
    )

    repaired = compile_code_repairs(src, output_path="pred_results/out.npy")

    assert "_mars_safe_map_array(data, mapping)" in repaired.source
    assert "os.makedirs('pred_results', exist_ok=True)" in repaired.source
    assert "totalize_numpy_dict_get_mapping" in repaired.repairs


def test_code_repair_compiler_rewrites_unique_manifest_basename(tmp_path: Path):
    benchmark = tmp_path / "benchmark"
    target = benchmark / "datasets" / "RealFolder" / "data.csv"
    target.parent.mkdir(parents=True)
    target.write_text("x\n1\n")
    src = "path = 'benchmark/datasets/WrongFolder/data.csv'\n"

    repaired = compile_code_repairs(src, benchmark_path=benchmark)

    assert "benchmark/datasets/RealFolder/data.csv" in repaired.source
    assert "rewrite_unique_dataset_basename" in repaired.repairs


def test_code_repair_compiler_rewrites_existing_extensionless_dataset_path(tmp_path: Path):
    benchmark = tmp_path / "benchmark"
    target = benchmark / "datasets" / "RealFolder" / "table"
    target.parent.mkdir(parents=True)
    target.write_text("x\n1\n")
    src = "path = 'benchmark/datasets/RealFolder/table.csv'\n"

    repaired = compile_code_repairs(src, benchmark_path=benchmark)

    assert "benchmark/datasets/RealFolder/table'" in repaired.source
    assert "rewrite_existing_extensionless_dataset_path" in repaired.repairs


def test_static_program_checks_find_unbounded_raster_read(tmp_path: Path):
    benchmark = tmp_path / "benchmark"
    (benchmark / "datasets").mkdir(parents=True)
    pred = tmp_path / "pred_raster.py"
    pred.write_text(
        "\n".join(
            [
                "import rasterio",
                "with rasterio.open('x.tif') as src:",
                "    data = src.read(1)",
            ]
        )
    )

    checks = _static_program_checks(pred_files=[pred], benchmark_path=benchmark)

    assert checks["n_unbounded_raster_reads"] == 1


def test_traceback_repair_removes_fit_epochs_keyword():
    src = "model.fit(X_train, y_train, epochs=50)\n"

    repaired = compile_code_repairs(
        src,
        failure_text="TypeError: TorchModel.fit() got an unexpected keyword argument 'epochs'",
    )

    assert "epochs=" not in repaired.source
    assert "remove_fit_epochs_keyword_on_signature_error" in repaired.repairs


def test_traceback_repair_coerces_mixed_dataframe_numeric_ops():
    src = "\n".join(
        [
            "import pandas as pd",
            "corr = data.corr(method='pearson')",
            "model.fit(X_train, y_train)",
        ]
    )

    repaired = compile_code_repairs(
        src,
        failure_text="ValueError: could not convert string to float: 'row_id'",
    )

    assert "data.select_dtypes(include='number').corr(" in repaired.source
    assert "model.fit(_mars_numeric_features(X_train), y_train)" in repaired.source
    assert "def _mars_numeric_features" in repaired.source
    assert "coerce_dataframe_numeric_features" in repaired.repairs


def test_traceback_repair_grounds_dataframe_column_access_without_breaking_assignment():
    src = "\n".join(
        [
            "train_counts['consensus_count'] = train_counts[['human counter 1', 'human counter 2']].mean(axis=1)",
            "y = df['classification']",
        ]
    )

    repaired = compile_code_repairs(
        src,
        failure_text="KeyError: None of [Index(['human counter 1'], dtype='object')] are in the [columns]",
    )

    assert "train_counts['consensus_count'] =" in repaired.source
    assert "_mars_columns(train_counts, ['human counter 1', 'human counter 2'])" in repaired.source
    assert "y = _mars_column(df, 'classification')" in repaired.source
    assert "def _mars_column(" in repaired.source
    assert "ground_dataframe_column_access" in repaired.repairs


def test_traceback_repair_adapts_deepchem_arrays_to_dataset():
    src = "\n".join(
        [
            "import deepchem as dc",
            "model.fit(X_train, y_train)",
            "predictions = model.predict(X_test)",
        ]
    )

    repaired = compile_code_repairs(
        src,
        failure_text=(
            "deepchem.models.torch_models.torch_model.py fit_generator "
            "TypeError: only integer scalar arrays can be converted to a scalar index"
        ),
    )

    assert "model.fit(dc.data.NumpyDataset(X_train, y_train))" in repaired.source
    assert "model.predict(dc.data.NumpyDataset(X_test))" in repaired.source
    assert "adapt_deepchem_arrays_to_numpy_dataset" in repaired.repairs


def test_traceback_repair_coerces_object_featurizer_outputs_to_numeric_tensor():
    src = "\n".join(
        [
            "import deepchem as dc",
            "X_train = featurizer.featurize(train_data['smiles'])",
            "X_test = featurizer.featurize(test_data['smiles'])",
        ]
    )

    repaired = compile_code_repairs(
        src,
        failure_text="TypeError: can't convert np.ndarray of type numpy.object_",
    )

    assert "X_train = _mars_numeric_tensor(featurizer.featurize(train_data['smiles']))" in repaired.source
    assert "X_test = _mars_numeric_tensor(featurizer.featurize(test_data['smiles']))" in repaired.source
    assert "def _mars_numeric_tensor" in repaired.source
    assert "coerce_object_feature_tensors" in repaired.repairs


def test_traceback_repair_numeric_tensor_is_idempotent():
    src = "X_train = _mars_numeric_tensor(featurizer.featurize(train_data['smiles']))\n"

    repaired = compile_code_repairs(
        src,
        failure_text="TypeError: can't convert np.ndarray of type numpy.object_",
    )

    assert "_mars_numeric_tensor(_mars_numeric_tensor" not in repaired.source
    assert "X_train = _mars_numeric_tensor(featurizer.featurize(train_data['smiles']))" in repaired.source


def test_traceback_repair_flattens_dataframe_column_arrays():
    src = "results = pd.DataFrame({'a': predictions[:, 0], 'b': predictions[:, 1]})\n"

    repaired = compile_code_repairs(
        src,
        failure_text="ValueError: Per-column arrays must each be 1-dimensional",
    )

    assert "'a': _mars_1d_column(predictions[:, 0])" in repaired.source
    assert "'b': _mars_1d_column(predictions[:, 1])" in repaired.source
    assert "def _mars_1d_column" in repaired.source
    assert "flatten_dataframe_column_arrays" in repaired.repairs
