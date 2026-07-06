"""Executable data/task contracts for weak-model program induction.

The compiler is intentionally benchmark-agnostic: it inspects files, output
paths, and evaluator source to produce a compact contract before asking an LLM
to write code.  The goal is to remove avoidable reasoning burden such as target
columns, train/test alignment, placeholder labels, and array shapes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import ast
import json
import re
from typing import Any


@dataclass(frozen=True)
class FileContract:
    path: str
    kind: str
    shape: tuple[int, ...] | tuple[int, int] | tuple[()]
    columns: tuple[str, ...] = ()
    numeric_columns: tuple[str, ...] = ()
    text_columns: tuple[str, ...] = ()
    constant_columns: tuple[str, ...] = ()
    placeholder_columns: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskContract:
    output_path: str
    output_kind: str
    dataset_files: tuple[FileContract, ...]
    evaluator_inputs: tuple[str, ...]
    evaluator_metrics: tuple[str, ...]
    success_thresholds: tuple[str, ...]
    required_output_columns: tuple[str, ...]
    target_candidates: tuple[str, ...]
    id_columns: tuple[str, ...]
    train_test_pairs: tuple[str, ...]
    array_shape_rules: tuple[str, ...]
    recommended_features: tuple[str, ...]
    preflight_checks: tuple[str, ...]
    scaffold_hints: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_prompt(self) -> str:
        return (
            "\nUNIVERSAL DATA/TASK CONTRACT (compiled from files and evaluator, not a hidden answer):\n"
            + json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
            + "\nCONTRACT-FIRST RULES:\n"
            "- Do not guess file schema. Use the paths, columns, shapes, and output contract above.\n"
            "- If train/test files share columns, build features from their shared non-id columns.\n"
            "- Never train on placeholder labels in test files; predict them.\n"
            "- Optimize the evaluator metric named in the contract; if the metric is ranking-based, save ranked scores.\n"
            "- Before saving, assert the required output columns/path/shape from the contract.\n"
            "- If execution fails, repair the violated contract first, then the scientific model.\n"
        )


def compile_task_contract(
    *,
    repo_root: str | Path,
    dataset_tree: str,
    output_path: str,
    eval_source: str = "",
    task_text: str = "",
    max_files: int = 16,
) -> TaskContract:
    root = Path(repo_root)
    dataset_files = _discover_dataset_files(root, dataset_tree, max_files=max_files)
    evaluator_inputs = _extract_paths(eval_source)
    evaluator_metrics = _extract_evaluator_metrics(eval_source)
    success_thresholds = _extract_success_thresholds(eval_source)
    required_output_columns = _extract_required_output_columns(eval_source, output_path)
    id_columns = _infer_id_columns(dataset_files, required_output_columns)
    target_candidates = _infer_targets(dataset_files, required_output_columns, task_text, eval_source)
    train_test_pairs = _infer_train_test_pairs(dataset_files)
    array_shape_rules = _infer_array_shape_rules(dataset_files, eval_source, output_path)
    recommended_features = _infer_recommended_features(dataset_files, target_candidates, id_columns)
    preflight_checks = _preflight_checks(
        output_path=output_path,
        output_kind=_kind_from_path(output_path),
        required_output_columns=required_output_columns,
        train_test_pairs=train_test_pairs,
        target_candidates=target_candidates,
        id_columns=id_columns,
        array_shape_rules=array_shape_rules,
    )
    scaffold_hints = _scaffold_hints(
        dataset_files=dataset_files,
        output_path=output_path,
        output_kind=_kind_from_path(output_path),
        evaluator_metrics=evaluator_metrics,
        required_output_columns=required_output_columns,
        target_candidates=target_candidates,
        id_columns=id_columns,
        recommended_features=recommended_features,
        array_shape_rules=array_shape_rules,
    )
    warnings = _warnings(dataset_files, target_candidates, train_test_pairs, required_output_columns)
    return TaskContract(
        output_path=output_path,
        output_kind=_kind_from_path(output_path),
        dataset_files=tuple(dataset_files),
        evaluator_inputs=tuple(evaluator_inputs),
        evaluator_metrics=tuple(evaluator_metrics),
        success_thresholds=tuple(success_thresholds),
        required_output_columns=tuple(required_output_columns),
        target_candidates=tuple(target_candidates),
        id_columns=tuple(id_columns),
        train_test_pairs=tuple(train_test_pairs),
        array_shape_rules=tuple(array_shape_rules),
        recommended_features=tuple(recommended_features),
        preflight_checks=tuple(preflight_checks),
        scaffold_hints=tuple(scaffold_hints),
        warnings=tuple(warnings),
    )


def task_contract_prompt(**kwargs: Any) -> str:
    return compile_task_contract(**kwargs).to_prompt()


def _discover_dataset_files(root: Path, dataset_tree: str, *, max_files: int) -> list[FileContract]:
    rels = _paths_from_tree(dataset_tree)
    files: list[Path] = []
    for rel in rels:
        p = root / "benchmark/datasets" / rel
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            files.extend(x for x in sorted(p.rglob("*")) if x.is_file())
    seen: set[Path] = set()
    contracts: list[FileContract] = []
    for p in files:
        if p in seen:
            continue
        seen.add(p)
        fc = _profile_file(root, p)
        if fc is not None:
            contracts.append(fc)
        if len(contracts) >= max_files:
            break
    return contracts


def _paths_from_tree(dataset_tree: str) -> list[str]:
    paths: list[str] = []
    stack: list[tuple[int, str]] = []
    for raw in dataset_tree.splitlines():
        match = re.match(r"\s*\|(-+)\s*(.+)", raw)
        if not match:
            continue
        depth = max(1, len(match.group(1)) // 2)
        name = match.group(2).strip().rstrip("/")
        if not name:
            continue
        while stack and stack[-1][0] >= depth:
            stack.pop()
        stack.append((depth, name))
        rel = "/".join(part for _, part in stack)
        paths.append(rel)
    return paths


def _profile_file(root: Path, path: Path) -> FileContract | None:
    rel = path.relative_to(root).as_posix()
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            import pandas as pd

            df = pd.read_csv(path)
            numeric = tuple(str(c) for c in df.select_dtypes(include="number").columns)
            text = tuple(str(c) for c in df.columns if str(c) not in numeric)
            constants = tuple(str(c) for c in df.columns if df[c].nunique(dropna=False) <= 1)
            placeholders = tuple(
                str(c) for c in df.columns
                if c in numeric and _looks_like_placeholder(df[c].dropna().tolist())
            )
            return FileContract(
                path=rel,
                kind="csv",
                shape=(int(df.shape[0]), int(df.shape[1])),
                columns=tuple(str(c) for c in df.columns),
                numeric_columns=numeric,
                text_columns=text,
                constant_columns=constants,
                placeholder_columns=placeholders,
            )
        if suffix == ".npy":
            import numpy as np

            arr = np.load(path, mmap_mode="r")
            return FileContract(path=rel, kind="npy", shape=tuple(int(x) for x in arr.shape))
        if suffix == ".npz":
            import numpy as np

            z = np.load(path)
            cols = tuple(str(k) for k in z.files)
            return FileContract(path=rel, kind="npz", shape=(len(cols),), columns=cols)
        if suffix in {".pkl", ".pickle"}:
            return FileContract(path=rel, kind="pickle", shape=())
    except Exception:
        return FileContract(path=rel, kind=f"{suffix[1:]}_unreadable", shape=())
    return FileContract(path=rel, kind=suffix[1:] or "file", shape=())


def _extract_paths(src: str) -> list[str]:
    paths = []
    for m in re.finditer(r"['\"]([^'\"]*(?:benchmark/datasets|pred_results|gold_results)[^'\"]*)['\"]", src):
        paths.append(m.group(1))
    return list(dict.fromkeys(paths))


def _extract_required_output_columns(eval_source: str, output_path: str) -> list[str]:
    cols: list[str] = []
    for var in _variables_reading_path(eval_source, output_path):
        pattern = rf"{re.escape(var)}\s*\[\s*['\"]([^'\"]+)['\"]\s*\]"
        cols.extend(m.group(1) for m in re.finditer(pattern, eval_source))
    return list(dict.fromkeys(cols))


def _extract_evaluator_metrics(src: str) -> list[str]:
    metrics: list[str] = []
    for name in ["roc_auc_score", "spearmanr", "accuracy_score", "f1_score", "mean_squared_error", "r2_score", "score_figure"]:
        if name in src:
            metrics.append(name)
    return metrics


def _extract_success_thresholds(src: str) -> list[str]:
    thresholds: list[str] = []
    for m in re.finditer(r"threshold\s*=\s*([0-9.]+)", src):
        thresholds.append(f"threshold={m.group(1)}")
    for m in re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)\s*>=\s*([0-9.]+)", src):
        thresholds.append(f"{m.group(1)}>={m.group(2)}")
    for m in re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)\s*==\s*([0-9]+)", src):
        thresholds.append(f"{m.group(1)}=={m.group(2)}")
    return list(dict.fromkeys(thresholds))


def _variables_reading_path(src: str, output_path: str) -> list[str]:
    vars_: list[str] = []
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return vars_
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        call = node.value
        if not isinstance(call, ast.Call):
            continue
        func = ast.unparse(call.func) if hasattr(ast, "unparse") else ""
        if not (func.endswith("read_csv") or func.endswith("load")):
            continue
        arg0 = call.args[0] if call.args else None
        if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
            if _same_tail_path(arg0.value, output_path):
                vars_.extend(t.id for t in node.targets if isinstance(t, ast.Name))
    return vars_


def _same_tail_path(a: str, b: str) -> bool:
    return a.strip("./") == b.strip("./") or a.strip("./").endswith(b.strip("./"))


def _infer_id_columns(files: list[FileContract], required_cols: list[str]) -> list[str]:
    names: set[str] = set()
    for col in required_cols:
        low = col.lower()
        if low in {"id", "index", "name", "sample", "sample_id"} or low.endswith("_id"):
            names.add(col)
    for fc in files:
        for col in fc.columns:
            low = col.lower()
            if low in {"id", "index", "name", "sample", "sample_id"} or low.endswith("_id"):
                names.add(col)
    return list(dict.fromkeys(names))


def _infer_targets(
    files: list[FileContract],
    required_cols: list[str],
    task_text: str,
    eval_source: str,
) -> list[str]:
    candidates: list[str] = []
    id_like = {"index", "id", "sample", "sample_id"}
    for col in required_cols:
        if col.lower() not in id_like:
            candidates.append(col)
    text = f"{task_text}\n{eval_source}".lower()
    for fc in files:
        for col in fc.columns:
            low = col.lower()
            if col in candidates:
                continue
            if low in id_like:
                continue
            if low in text or low.replace("-", " ") in text:
                candidates.append(col)
            elif col in fc.placeholder_columns:
                candidates.append(col)
    return list(dict.fromkeys(candidates))


def _infer_train_test_pairs(files: list[FileContract]) -> list[str]:
    pairs: list[str] = []
    trains = [f for f in files if "/train/" in f.path or "train" in Path(f.path).name.lower()]
    tests = [f for f in files if "/test/" in f.path or "test" in Path(f.path).name.lower()]
    for tr in trains:
        for te in tests:
            if tr.kind != te.kind:
                continue
            if tr.kind == "csv":
                shared = [c for c in tr.columns if c in te.columns]
                if shared:
                    pairs.append(
                        f"{tr.path} -> {te.path}; shared_columns={','.join(shared[:24])}"
                    )
            elif tr.kind in {"npy", "npz"}:
                pairs.append(f"{tr.path} -> {te.path}; train_shape={tr.shape}; test_shape={te.shape}")
    return pairs


def _infer_array_shape_rules(files: list[FileContract], eval_source: str, output_path: str) -> list[str]:
    rules: list[str] = []
    for m in re.finditer(r"reshape\([^,\n]+,\s*\[([^\]]+)\]\)", eval_source):
        dims = "x".join(x.strip() for x in m.group(1).split(","))
        rules.append(f"evaluator reshapes reference to [{dims}]; saved array should be comparable when flattened")
    for fc in files:
        if fc.kind == "npy":
            rules.append(f"{fc.path} shape={fc.shape}")
    if output_path.endswith(".npy") and any("evaluator reshapes" in r for r in rules):
        rules.append("save a numeric .npy array, not CSV/text; avoid changing sample order")
    return list(dict.fromkeys(rules))


def _infer_recommended_features(
    files: list[FileContract],
    targets: list[str],
    id_columns: list[str],
) -> list[str]:
    csvs = [f for f in files if f.kind == "csv"]
    trains = [f for f in csvs if "train" in Path(f.path).name.lower() or "/train/" in f.path]
    tests = [f for f in csvs if "test" in Path(f.path).name.lower() or "/test/" in f.path]
    if not trains or not tests:
        return []
    shared = set(tests[0].numeric_columns)
    for te in tests[1:]:
        shared &= set(te.numeric_columns)
    for tr in trains:
        shared &= set(tr.numeric_columns)
    banned = set(targets) | set(id_columns)
    for fc in tests:
        banned |= set(fc.placeholder_columns)
    return sorted(c for c in shared if c not in banned)


def _preflight_checks(
    *,
    output_path: str,
    output_kind: str,
    required_output_columns: list[str],
    train_test_pairs: list[str],
    target_candidates: list[str],
    id_columns: list[str],
    array_shape_rules: list[str],
) -> list[str]:
    checks = [f"create parent directory and save exactly to {output_path}"]
    if output_kind == "csv" and required_output_columns:
        checks.append("CSV output columns must include, in order if possible: " + ",".join(required_output_columns))
    if target_candidates:
        checks.append("exclude target columns from X/features: " + ",".join(target_candidates))
    if id_columns:
        checks.append("preserve id/index columns for output alignment: " + ",".join(id_columns))
    if train_test_pairs:
        checks.append("align train/test by shared feature columns before fitting or transforming")
    if array_shape_rules:
        checks.append("for array outputs, assert saved array shape is compatible with evaluator reshape/flatten rules")
    return checks


def _scaffold_hints(
    *,
    dataset_files: list[FileContract],
    output_path: str,
    output_kind: str,
    evaluator_metrics: list[str],
    required_output_columns: list[str],
    target_candidates: list[str],
    id_columns: list[str],
    recommended_features: list[str],
    array_shape_rules: list[str],
) -> list[str]:
    hints: list[str] = []
    if output_kind == "csv" and target_candidates and recommended_features:
        train = next(
            (f for f in dataset_files if f.kind == "csv" and ("train" in Path(f.path).name.lower() or "/train/" in f.path)),
            None,
        )
        test = next(
            (f for f in dataset_files if f.kind == "csv" and ("test" in Path(f.path).name.lower() or "/test/" in f.path)),
            None,
        )
        if train and test:
            target = target_candidates[0]
            id_col = id_columns[0] if id_columns else required_output_columns[0] if required_output_columns else ""
            features = ", ".join(repr(c) for c in recommended_features[:40])
            cols = ", ".join(repr(c) for c in required_output_columns) if required_output_columns else ""
            hints.append(
                "CSV scaffold: "
                f"train=pd.read_csv('{train.path}'); test=pd.read_csv('{test.path}'); "
                f"target_col={target!r}; id_col={id_col!r}; feature_cols=[{features}]; "
                "X_train=train[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0); "
                "X_test=test[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0); "
                "fit only on train[target_col], never on test[target_col]; "
                f"save DataFrame with columns [{cols}] to {output_path!r}."
            )
            hints.append(
                "If the task asks for binary classification from a continuous target, derive y from train[target_col] "
                "using a stated/domain threshold when present, otherwise a data-driven threshold, and save a score or "
                "probability for the target column when the evaluator uses ranking/AUC."
            )
            if "roc_auc_score" in evaluator_metrics:
                hints.append(
                    "Metric scaffold: evaluator uses roc_auc_score, so output the positive-class score/probability "
                    "in the target column when possible; hard 0/1 labels are allowed but usually give weaker ranking."
                )
    if output_kind == "npy" and array_shape_rules:
        npy_files = [f for f in dataset_files if f.kind == "npy"]
        if npy_files:
            hints.append(
                "NPY scaffold: load arrays with np.load; standardize train source and target with train statistics; "
                "reshape paired arrays consistently before fitting; apply the same source transform to test; "
                f"save numeric array with np.save({output_path!r}, arr)."
            )
            hints.append(
                "For EEG-like arrays shaped (channels, samples, time), a common contract-safe transform is "
                "np.transpose(arr, (1, 0, 2)).reshape(n_samples, channels*time); infer n_samples from the array shape."
            )
            hints.append(
                "Budget scaffold: prefer a fast linear/Ridge/least-squares baseline on reshaped arrays before any "
                "deep training loop; use subsampling if needed and keep runtime bounded."
            )
            if "spearmanr" in evaluator_metrics:
                hints.append(
                    "Metric scaffold: evaluator uses Spearman correlation on flattened arrays, so preserve sample order "
                    "and focus on rank-consistent numeric outputs, not only exact scale."
                )
    return hints


def _warnings(
    files: list[FileContract],
    targets: list[str],
    train_test_pairs: list[str],
    required_output_columns: list[str],
) -> list[str]:
    warnings: list[str] = []
    if any(fc.placeholder_columns for fc in files):
        warnings.append("placeholder_columns_detected_do_not_use_as_observed_test_labels")
    if required_output_columns and not targets:
        warnings.append("output_columns_found_but_no_non_id_target_candidate")
    if not train_test_pairs and len(files) > 1:
        warnings.append("no_clear_train_test_pair_detected")
    return warnings


def _looks_like_placeholder(values: list[Any]) -> bool:
    if not values:
        return False
    uniq = {float(v) for v in values if isinstance(v, (int, float))}
    return len(uniq) == 1 and next(iter(uniq)) in {-999.0, -1.0, 999.0}


def _kind_from_path(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".npy":
        return "npy"
    if suffix == ".npz":
        return "npz"
    if suffix in {".png", ".jpg", ".jpeg"}:
        return "image"
    if suffix == ".txt":
        return "text"
    return suffix[1:] or "file"
