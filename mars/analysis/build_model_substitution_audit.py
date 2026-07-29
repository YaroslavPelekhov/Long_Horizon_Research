"""Validate and summarize complete proposal-core substitution runs."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "paper_assets" / "evidence"

DISCOVERY = (
    (
        "GPT-4o-mini",
        "lmw/universal_discovery_real/"
        "aaai27_language_growth_protocol_v1_fixed_l0_test239/summary.json",
    ),
    (
        "Gemini 2.5 Flash Lite",
        "lmw/universal_discovery_real/"
        "aaai27_modelcore_gemini_flash_lite_discovery_full239_matched_20260729/"
        "summary.json",
    ),
    (
        "Qwen3-30B-A3B-Instruct",
        "lmw/universal_discovery_real/"
        "aaai27_modelcore_qwen3_30b_discovery_full239_matched_20260729/"
        "summary.json",
    ),
)

NEWTON = (
    (
        "GPT-4o-mini",
        "lmw/nb_activeprobe/"
        "aaai27_newton_clean_20260724_seed2031_seed2031/summary.json",
    ),
    (
        "Gemini 2.5 Flash Lite",
        "lmw/nb_activeprobe/"
        "aaai27_modelcore_gemini_flash_lite_newton_full324_20260721/summary.json",
    ),
    (
        "Qwen3-30B-A3B-Instruct",
        "lmw/nb_activeprobe/"
        "aaai27_modelcore_qwen3_30b_newton_full324_20260721/summary.json",
    ),
    (
        "GPT-4o",
        "lmw/nb_activeprobe/"
        "aaai27_modelcore_gpt4o_newton_full324_rghli_20260721/summary.json",
    ),
    (
        "DeepSeek V4 Flash",
        "lmw/nb_activeprobe/"
        "aaai27_modelcore_deepseek_v4_flash_newton_full324_rghli_20260721/"
        "summary.json",
    ),
)

ULTRAHORIZON = (
    (
        "GPT-4o-mini",
        {
            env: f"lmw/uh_official/uh_clean_universal_{env}_full32_20260715/"
            "summary.json"
            for env in ("grid", "seq", "bio")
        },
    ),
    (
        "Gemini 2.5 Flash Lite",
        {
            env: f"lmw/uh_official/aaai27_modelcore_gemini_uh_{env}_"
            "full32_20260721/summary.json"
            for env in ("grid", "seq", "bio")
        },
    ),
    (
        "Qwen3-30B-A3B-Instruct",
        {
            env: f"lmw/uh_official/aaai27_modelcore_qwen_uh_{env}_"
            "full32_20260721/summary.json"
            for env in ("grid", "seq", "bio")
        },
    ),
)


def _read(relative_path: str) -> tuple[Path, dict[str, Any]]:
    path = ROOT / relative_path
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return path, value


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _discovery_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, relative_path in DISCOVERY:
        path, value = _read(relative_path)
        expected = {
            "data_split": "test",
            "n_tasks": 239,
            "n_proposals": 4,
            "max_rounds": 1,
            "judge_model": "openai/gpt-4o",
        }
        for key, target in expected.items():
            if value.get(key) != target:
                raise ValueError(f"Discovery mismatch for {model}: {key}")
        rows.append(
            {
                "benchmark": "DiscoveryBench",
                "protocol": "matched frozen 4-proposal/1-round",
                "model": model,
                "n": 239,
                "primary_metric": "HMS",
                "primary": float(value["HMS_mean_100"]),
                "secondary_metric": "Cons-HMS",
                "secondary": float(value["HMS_mean_consistency_100"]),
                "source": relative_path,
                "sha256": _digest(path),
            }
        )
    return rows


def _newton_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    protocol_keys = ("modules", "difficulties", "law_versions", "systems")
    reference: dict[str, Any] | None = None
    for model, relative_path in NEWTON:
        path, value = _read(relative_path)
        if int(value.get("n", -1)) != 324:
            raise ValueError(f"NewtonBench is not complete for {model}")
        protocol = {key: value.get(key) for key in protocol_keys}
        if reference is None:
            reference = protocol
        elif protocol != reference:
            raise ValueError(f"NewtonBench protocol mismatch for {model}")
        answered = int(value["answered"])
        correct = round(float(value["SA_all"]) * 324)
        rows.append(
            {
                "benchmark": "NewtonBench",
                "protocol": "frozen diagnostic kernel",
                "model": model,
                "n": 324,
                "answered": answered,
                "correct": correct,
                "primary_metric": "SA-all",
                "primary": 100.0 * correct / 324,
                "secondary_metric": "SA-answered",
                "secondary": 100.0 * correct / answered,
                "source": relative_path,
                "sha256": _digest(path),
            }
        )
    return rows


def _ultrahorizon_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, paths in ULTRAHORIZON:
        scores: dict[str, float] = {}
        sources: list[str] = []
        digests: list[str] = []
        for env, relative_path in paths.items():
            path, value = _read(relative_path)
            expected = {
                "n": 32,
                "steps": 50,
                "difficulty": "hard",
                "action_budget": 220,
                "paper_style_judge": True,
                "exact_paper_judge": True,
                "use_env_hints": False,
                "enable_fallback_commit": True,
            }
            for key, target in expected.items():
                if value.get(key) != target:
                    raise ValueError(
                        f"UltraHorizon mismatch for {model}/{env}: {key}"
                    )
            scores[env] = float(value["mean_score"])
            sources.append(relative_path)
            digests.append(_digest(path))
        rows.append(
            {
                "benchmark": "UltraHorizon",
                "protocol": "controller-enabled exact paper judge",
                "model": model,
                "n": 96,
                "grid": scores["grid"],
                "seq": scores["seq"],
                "bio": scores["bio"],
                "primary_metric": "environment score",
                "primary": sum(scores.values()) / 3.0,
                "secondary_metric": "",
                "secondary": "",
                "source": ";".join(sources),
                "sha256": ";".join(digests),
            }
        )
    return rows


def main() -> None:
    rows = _discovery_rows() + _newton_rows() + _ultrahorizon_rows()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "model_substitution_audit.csv"
    json_path = OUTPUT_DIR / "model_substitution_audit.json"

    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(
        json.dumps(
            {
                "audit_version": "aaai27-model-substitution-v1",
                "protocols_validated": True,
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(rows)} validated rows")


if __name__ == "__main__":
    main()
