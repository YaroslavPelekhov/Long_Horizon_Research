"""Preflight and run ScienceAgentBench official Docker evaluation.

This wrapper keeps SAB scoring honest: it refuses to report proxy metrics as
official SR/VER/CBS and records infrastructure blockers separately.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_PROJ = Path(__file__).resolve().parent.parent.parent
if str(_PROJ) not in sys.path:
    sys.path.insert(0, str(_PROJ))

try:
    from dotenv import load_dotenv

    env_path = _PROJ / "autodiscovery" / ".env.local"
    if env_path.exists():
        load_dotenv(env_path, override=True)
except ImportError:
    pass


REQUIRED_BENCHMARK_DIRS = [
    "datasets",
    "eval_programs",
    "gold_programs",
    "scoring_rubrics",
]

_DATASET_REF_RE = re.compile(
    r"benchmark/datasets/[A-Za-z0-9_./+\-]+"
)


def _static_program_checks(
    *, pred_files: list[Path], benchmark_path: Path
) -> dict[str, Any]:
    """Cheap universal checks for generated SAB programs.

    These checks are intentionally not benchmark-specific.  They catch two
    recurring failure modes of self-written scientific code before we spend
    minutes building Docker images:

    1. references to dataset files that are not present in the benchmark tree;
    2. unsafe category maps that use ``dict.get`` without a default value.
    """

    missing_dataset_refs: list[dict[str, str]] = []
    unsafe_mapping_get: list[dict[str, str]] = []
    unbounded_raster_reads: list[dict[str, str]] = []
    datasets_root = benchmark_path / "datasets"

    for pred in pred_files:
        text = pred.read_text(encoding="utf-8", errors="ignore")
        for ref in sorted(set(_DATASET_REF_RE.findall(text))):
            rel = ref.removeprefix("benchmark/datasets/").rstrip(").,;:'\"]")
            if rel and not (datasets_root / rel).exists():
                missing_dataset_refs.append(
                    {"program": pred.name, "ref": ref, "expected": str(datasets_root / rel)}
                )

        for lineno, line in enumerate(text.splitlines(), 1):
            compact = line.replace(" ", "")
            if "np.vectorize(" in compact and ".get)" in compact:
                unsafe_mapping_get.append(
                    {
                        "program": pred.name,
                        "line": str(lineno),
                        "pattern": "np.vectorize(dict.get) without explicit default",
                    }
                )
            if "rasterio.open" in text and ".read(1)" in compact and "block_windows" not in text:
                unbounded_raster_reads.append(
                    {
                        "program": pred.name,
                        "line": str(lineno),
                        "pattern": "whole-raster read without window/block iteration",
                    }
                )

    return {
        "n_missing_dataset_refs": len(missing_dataset_refs),
        "missing_dataset_refs": missing_dataset_refs[:40],
        "n_unsafe_mapping_get": len(unsafe_mapping_get),
        "unsafe_mapping_get": unsafe_mapping_get[:40],
        "n_unbounded_raster_reads": len(unbounded_raster_reads),
        "unbounded_raster_reads": unbounded_raster_reads[:40],
        "n_static_warnings": (
            len(missing_dataset_refs)
            + len(unsafe_mapping_get)
            + len(unbounded_raster_reads)
        ),
    }


def _dataset_coverage(benchmark_path: Path, *, split: str = "verified") -> dict[str, Any]:
    datasets_root = benchmark_path / "datasets"
    local_folders = {
        p.name for p in datasets_root.iterdir()
        if p.is_dir()
    } if datasets_root.exists() else set()
    try:
        from datasets import load_dataset

        ds = load_dataset("osunlp/ScienceAgentBench", split=split)
        required: list[dict[str, str]] = []
        for row in ds:
            tree = str(row.get("dataset_folder_tree") or "")
            first = tree.split("\n")[0] if tree else ""
            folder = first.replace("|-- ", "").strip().rstrip("/")
            if folder:
                required.append(
                    {
                        "instance_id": str(row.get("instance_id", "")),
                        "folder": folder,
                        "gold_program_name": str(row.get("gold_program_name", "")),
                    }
                )
        missing = [row for row in required if row["folder"] not in local_folders]
        return {
            "status": "ok",
            "split": split,
            "n_required_tasks": len(required),
            "n_local_top_folders": len(local_folders),
            "n_missing_top_folders": len(missing),
            "missing_top_folders": missing[:60],
        }
    except Exception as exc:
        return {
            "status": "unavailable",
            "error": f"{type(exc).__name__}: {exc}",
            "n_local_top_folders": len(local_folders),
        }


def _run(args: list[str], *, cwd: Path, timeout: int | None = None) -> tuple[int, str]:
    try:
        p = subprocess.run(
            args,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
        return p.returncode, p.stdout
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="ignore")
        return 124, f"TimeoutExpired after {timeout}s\n{output}"
    except Exception as exc:
        return 999, f"{type(exc).__name__}: {exc}"


def _jsonl_scores(path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return {
        "n_eval_rows": len(rows),
        "rows": rows[:20],
    }


def _preflight(
    *,
    repo: Path,
    benchmark_path: Path,
    pred_program_path: Path,
    split: str = "verified",
) -> dict[str, Any]:
    blockers: list[str] = []

    required_missing = [
        name for name in REQUIRED_BENCHMARK_DIRS if not (benchmark_path / name).exists()
    ]
    if required_missing:
        blockers.append(
            "missing benchmark_verified artifacts: " + ", ".join(required_missing)
        )

    static_checks: dict[str, Any] = {
        "n_missing_dataset_refs": 0,
        "missing_dataset_refs": [],
        "n_unsafe_mapping_get": 0,
        "unsafe_mapping_get": [],
        "n_unbounded_raster_reads": 0,
        "unbounded_raster_reads": [],
        "n_static_warnings": 0,
    }

    if not pred_program_path.exists():
        blockers.append(f"pred_program_path does not exist: {pred_program_path}")
        pred_files: list[Path] = []
    else:
        pred_files = sorted(pred_program_path.glob("pred_*.py"))
        if not pred_files:
            blockers.append(f"pred_program_path has no pred_*.py files: {pred_program_path}")
        bad_preds = []
        for pred in pred_files:
            try:
                text = pred.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                text = ""
            if pred.stat().st_size < 100 or text.strip() == "ERROR":
                bad_preds.append(pred.name)
        if bad_preds:
            blockers.append("invalid prediction files: " + ", ".join(bad_preds[:12]))
        static_checks = _static_program_checks(
            pred_files=pred_files,
            benchmark_path=benchmark_path,
        )

    docker_code, docker_out = _run(["docker", "info"], cwd=repo, timeout=12)
    if docker_code != 0:
        blockers.append("Docker daemon is not reachable")

    openai_key = os.environ.get("OPENAI_API_KEY", "")
    azure_ok = all(
        os.environ.get(k)
        for k in [
            "AZURE_OPENAI_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_VERSION",
            "AZURE_OPENAI_DEPLOYMENT_NAME",
        ]
    )
    if not openai_key and not azure_ok:
        blockers.append("missing direct OpenAI or Azure credentials")

    openai_base = os.environ.get("OPENAI_BASE_URL", "")
    openrouter_like = bool(openai_key.startswith("sk" + "-or-") or "openrouter" in openai_base)
    if openrouter_like and not os.environ.get("OPENAI_VISUAL_JUDGE_MODEL"):
        os.environ["OPENAI_VISUAL_JUDGE_MODEL"] = "openai/gpt-4o"

    return {
        "blockers": blockers,
        "docker_info_ok": docker_code == 0,
        "docker_info_output_head": docker_out[:1200],
        "benchmark_path": str(benchmark_path),
        "pred_program_path": str(pred_program_path),
        "n_pred_files": len(pred_files),
        "dataset_coverage": _dataset_coverage(benchmark_path, split=split),
        "static_program_checks": static_checks,
        "required_missing": required_missing,
        "openai_key_present": bool(openai_key),
        "openai_base_url_present": bool(openai_base),
        "openrouter_like_route": openrouter_like,
        "visual_judge_model": os.environ.get("OPENAI_VISUAL_JUDGE_MODEL", "gpt-4o-2024-05-13"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", default="sab_official_eval")
    parser.add_argument("--pred_program_path", default="")
    parser.add_argument("--benchmark_path", default="")
    parser.add_argument("--log_fname", default="")
    parser.add_argument("--instance_ids", nargs="*", default=None)
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument(
        "--instance_timeout",
        type=int,
        default=1800,
        help="Official harness timeout in seconds for each SAB instance.",
    )
    parser.add_argument(
        "--harness_timeout",
        type=int,
        default=0,
        help="Wrapper timeout in seconds for the whole official harness command; 0 disables.",
    )
    parser.add_argument("--split", default="verified")
    parser.add_argument("--cache_level", default="base")
    parser.add_argument("--force_rebuild", action="store_true")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--preflight_only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repo = _PROJ / "scienceagentbench_repo"
    out_dir = _PROJ / "lmw" / "sab_official_eval" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    if summary_path.exists() and not args.overwrite:
        raise SystemExit(f"summary exists; pass --overwrite: {summary_path}")

    pred_program_path = (
        Path(args.pred_program_path)
        if args.pred_program_path
        else _PROJ / "lmw" / "sab_official" / "sab_export_verified4_gpt4omini_v1" / "pred_programs"
    ).resolve()
    benchmark_path = (Path(args.benchmark_path) if args.benchmark_path else repo / "benchmark").resolve()
    log_fname = Path(args.log_fname) if args.log_fname else out_dir / "eval.jsonl"
    if log_fname.exists() and args.overwrite:
        log_fname.unlink()

    preflight = _preflight(
        repo=repo,
        benchmark_path=benchmark_path,
        pred_program_path=pred_program_path,
        split=args.split,
    )
    summary: dict[str, Any] = {
        "run_id": args.run_id,
        "score_type": "official-sab-docker" if not preflight["blockers"] else "blocked-preflight",
        "preflight": preflight,
        "eval_log": str(log_fname),
        "started_at": time.time(),
    }

    if preflight["blockers"] or args.preflight_only:
        summary["official_eval_skipped"] = True
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print("=== SAB official eval preflight ===")
        print(f"summary={summary_path}")
        if preflight["blockers"]:
            print("blockers:")
            for b in preflight["blockers"]:
                print(f"  - {b}")
        else:
            print("preflight ok")
        return

    cmd = [
        sys.executable,
        "-m",
        "evaluation.harness.run_evaluation",
        "--benchmark_path",
        str(benchmark_path),
        "--pred_program_path",
        str(pred_program_path),
        "--log_fname",
        str(log_fname),
        "--run_id",
        args.run_id,
        "--split",
        args.split,
        "--max_workers",
        str(args.max_workers),
        "--cache_level",
        args.cache_level,
        "--timeout",
        str(args.instance_timeout),
    ]
    if args.instance_ids:
        cmd += ["--instance_ids", *[str(x) for x in args.instance_ids]]
    if args.force_rebuild:
        cmd.append("--force_rebuild")
    if args.clean:
        cmd.append("--clean")

    print("=== SAB official Docker eval ===")
    print(" ".join(cmd))
    code, output = _run(cmd, cwd=repo, timeout=args.harness_timeout or None)
    summary.update(
        {
            "official_eval_skipped": False,
            "returncode": code,
            "harness_output_tail": output[-5000:],
            "eval": _jsonl_scores(log_fname),
            "finished_at": time.time(),
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"returncode={code}")
    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
