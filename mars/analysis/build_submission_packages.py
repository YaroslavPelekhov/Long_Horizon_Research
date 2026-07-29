"""Build clean, compile-verified AAAI submission and anonymous code archives."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[2]
SECRET_PATTERNS = (
    re.compile(rb"sk-or-v1-[A-Za-z0-9_-]{20,}"),
    re.compile(rb"sk-[A-Za-z0-9_-]{40,}"),
)

MAIN_FIGURES = (
    "hypothesis_language_growth_pipeline.png",
    "newton_coverage_growth_composite.pdf",
)
SUPPLEMENT_FIGURES = (
    "concrete_language_growth_step.png",
    "language_growth_audit.pdf",
)
CODE_TESTS = (
    "test_answer_contract.py",
    "test_answer_plan_inducer.py",
    "test_answer_slot_compiler.py",
    "test_contract_baselines.py",
    "test_estimand_synthesizer.py",
    "test_evidence_contract_compiler.py",
    "test_interface_profiler.py",
    "test_main_experiment_protocols.py",
    "test_metamorphic_estimand_kernel.py",
    "test_metric_compiler.py",
    "test_model_substitution_audit.py",
    "test_nb_activeprobe_resume.py",
    "test_newton_code_executor_timeout.py",
    "test_operator_genome.py",
    "test_paper_evidence_index.py",
    "test_problem_frame_inducer.py",
    "test_program_induction_ledger.py",
    "test_residual_class_ledger.py",
    "test_residual_kernel.py",
    "test_scope_abstraction.py",
    "test_self_induced_language.py",
    "test_skill_grammar.py",
    "test_slot_contract.py",
    "test_submission_packaging.py",
    "test_task_contract.py",
    "test_typed_operator_plan.py",
    "test_universal_cpi_seed.py",
    "test_universal_discovery_real_eval.py",
    "test_universal_hypothesis_kernel.py",
    "test_universal_slot_compiler.py",
)
CODE_RUNNERS = (
    "__init__.py",
    "run_db_official_eval.py",
    "run_discovery_chunked_eval.py",
    "run_discovery_external_baselines.py",
    "run_language_growth_experiment.py",
    "run_nb_activeprobe.py",
    "run_nb_fitters.py",
    "run_nb_official_dpsr.py",
    "run_nb_selfverify.py",
    "run_newton_clean_multirun.py",
    "run_newton_language_growth_trajectory.py",
    "run_newton_language_transfer.py",
    "run_uh_official.py",
    "run_universal_discovery_real_eval.py",
    "summarize_language_transfer.py",
)


def _copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _copy_python_tree(source: Path, target: Path) -> None:
    for path in source.rglob("*.py"):
        _copy(path, target / path.relative_to(source))


def _copy_reviewer_code(destination: Path) -> None:
    for name in ("__init__.py", "coordinator.py", "smoketest.py"):
        _copy(ROOT / "mars" / name, destination / "mars" / name)
    for subdir in ("adapters", "agents", "analysis", "induction", "mdl", "skills"):
        source = ROOT / "mars" / subdir
        for path in source.rglob("*.py"):
            if path.name in {
                "scienceagentbench_adapter.py",
                "build_submission_ablation_results.py",
            }:
                continue
            _copy(path, destination / "mars" / path.relative_to(ROOT / "mars"))
    for name in CODE_RUNNERS:
        _copy(
            ROOT / "mars" / "runners" / name,
            destination / "mars" / "runners" / name,
        )
    _copy(ROOT / "tests" / "__init__.py", destination / "tests" / "__init__.py")
    for name in CODE_TESTS:
        _copy(ROOT / "tests" / name, destination / "tests" / name)


def _compile(directory: Path, expected_pages: int | None = None) -> None:
    env = os.environ.copy()
    env.update({"LC_ALL": "C", "LANG": "C"})
    command = [
        "latexmk",
        "-pdf",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "main.tex",
    ]
    subprocess.run(
        command,
        cwd=directory,
        env=env,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    pdf = directory / "main.pdf"
    if expected_pages is not None and len(PdfReader(pdf).pages) != expected_pages:
        raise ValueError(
            f"{directory.name} compiled to {len(PdfReader(pdf).pages)} pages; "
            f"expected {expected_pages}"
        )


def _secret_scan(directory: Path) -> None:
    identity_patterns = (
        str(ROOT).encode(),
        str(Path.home()).encode(),
        Path.home().name.encode(),
    )
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        data = path.read_bytes()
        if any(pattern.search(data) for pattern in SECRET_PATTERNS):
            raise ValueError(f"credential-like string found in package: {path}")
        if any(pattern and pattern in data for pattern in identity_patterns):
            raise ValueError(f"local identity or path found in package: {path}")


def _anonymize_local_paths(directory: Path) -> None:
    replacements = (
        ((str(ROOT) + os.sep).encode(), b""),
        (str(ROOT).encode(), b"."),
        ((str(Path.home()) + os.sep).encode(), b"$HOME/"),
        (str(Path.home()).encode(), b"$HOME"),
    )
    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        original = path.read_bytes()
        sanitized = original
        for old, new in replacements:
            sanitized = sanitized.replace(old, new)
        if sanitized != original:
            path.write_bytes(sanitized)


def _rebuild_packaged_evidence(destination: Path) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(destination)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    for module in (
        "mars.analysis.build_model_substitution_audit",
        "mars.analysis.build_paper_evidence_index",
    ):
        subprocess.run(
            [sys.executable, "-m", module],
            cwd=destination,
            env=env,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    for cache in destination.rglob("__pycache__"):
        shutil.rmtree(cache)


def _zip(directory: Path) -> Path:
    archive = directory.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(directory))
    return archive


def _clean_latex_build_files(directory: Path) -> None:
    keep = {".pdf", ".tex", ".bib", ".sty", ".bst", ".png", ".csv", ".json", ".md"}
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix not in keep:
            path.unlink()


def _copy_indexed_evidence(destination: Path) -> None:
    index_path = ROOT / "paper_assets" / "evidence" / "evidence_index.json"
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    for entry in payload["entries"]:
        relative = Path(entry["artifact"])
        source = (ROOT / relative).resolve()
        if not source.is_relative_to(ROOT.resolve()):
            raise ValueError(f"evidence path escapes repository root: {relative}")
        if not source.is_file():
            raise FileNotFoundError(source)
        _copy(source, destination / relative)


def _build_main(destination: Path) -> None:
    _copy(ROOT / "residual_guided_hli_submission.tex", destination / "main.tex")
    for name in ("residual_guided_hli_refs.bib", "aaai2027.sty", "aaai2027.bst"):
        _copy(ROOT / name, destination / name)
    for name in MAIN_FIGURES:
        _copy(
            ROOT / "paper_assets" / "figures" / name,
            destination / "paper_assets" / "figures" / name,
        )
    _compile(destination, expected_pages=9)
    _clean_latex_build_files(destination)


def _build_supplement(destination: Path) -> None:
    _copy(ROOT / "residual_guided_hli_supplement.tex", destination / "main.tex")
    for name in ("aaai2027.sty", "aaai2027.bst"):
        _copy(ROOT / name, destination / name)
    for name in SUPPLEMENT_FIGURES:
        _copy(
            ROOT / "paper_assets" / "figures" / name,
            destination / "paper_assets" / "figures" / name,
        )
    for name in (
        "README.md",
        "evidence_index.csv",
        "evidence_index.json",
        "model_substitution_audit.csv",
        "model_substitution_audit.json",
    ):
        _copy(
            ROOT / "paper_assets" / "evidence" / name,
            destination / "paper_assets" / "evidence" / name,
        )
    _compile(destination, expected_pages=8)
    _clean_latex_build_files(destination)


def _build_checklist(destination: Path) -> None:
    _copy(ROOT / "reproducibility_checklist_main.tex", destination / "main.tex")
    _copy(ROOT / "ReproducibilityChecklist.tex", destination / "ReproducibilityChecklist.tex")
    _copy(ROOT / "aaai2027.sty", destination / "aaai2027.sty")
    _compile(destination, expected_pages=2)
    _clean_latex_build_files(destination)


def _build_code(destination: Path) -> None:
    _copy_reviewer_code(destination)
    _copy_python_tree(ROOT / "openai", destination / "openai")
    if (ROOT / "ols").is_dir():
        _copy_python_tree(ROOT / "ols", destination / "ols")
    _copy(ROOT / "ANONYMOUS_CODE_RELEASE.md", destination / "README.md")
    _copy(ROOT / "REPRODUCIBILITY.md", destination / "REPRODUCIBILITY.md")
    _copy(ROOT / "RESULTS.md", destination / "RESULTS.md")
    _copy(ROOT / "LICENSE", destination / "LICENSE")
    _copy(ROOT / "pytest.ini", destination / "pytest.ini")
    _copy(ROOT / "requirements-paper.txt", destination / "requirements.txt")
    for name in (
        "README.md",
        "evidence_index.csv",
        "evidence_index.json",
        "model_substitution_audit.csv",
        "model_substitution_audit.json",
    ):
        _copy(
            ROOT / "paper_assets" / "evidence" / name,
            destination / "paper_assets" / "evidence" / name,
        )
    _copy_indexed_evidence(destination)
    for subdir in ("newton_matched_controls", "newton_language_growth", "headline_manifest"):
        source = ROOT / "paper_assets" / "evidence" / subdir
        for path in source.rglob("*"):
            if path.is_file():
                _copy(path, destination / path.relative_to(ROOT))
    _anonymize_local_paths(destination)
    _rebuild_packaged_evidence(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=False)

    builders = {
        "overleaf_main": _build_main,
        "overleaf_supplement": _build_supplement,
        "reproducibility_checklist": _build_checklist,
        "anonymous_code_release": _build_code,
    }
    for name, builder in builders.items():
        destination = output_root / name
        destination.mkdir()
        builder(destination)
        _secret_scan(destination)
        archive = _zip(destination)
        print(f"{name}: {archive}")


if __name__ == "__main__":
    main()
