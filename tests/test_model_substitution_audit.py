from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "paper_assets" / "evidence" / "model_substitution_audit.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_model_substitution_audit_is_complete_and_source_hashed() -> None:
    payload = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    rows = payload["rows"]

    assert payload["protocols_validated"] is True
    assert len(rows) == 11
    assert Counter(row["benchmark"] for row in rows) == {
        "DiscoveryBench": 3,
        "NewtonBench": 5,
        "UltraHorizon": 3,
    }

    missing = [
        source
        for row in rows
        for source in row["source"].split(";")
        if not (ROOT / source).is_file()
    ]
    if missing:
        pytest.skip("full evaluator artifact bundle is not mounted")

    for row in rows:
        sources = row["source"].split(";")
        digests = row["sha256"].split(";")
        assert len(sources) == len(digests)
        for source, digest in zip(sources, digests, strict=True):
            path = ROOT / source
            assert path.is_file()
            assert digest == _sha256(path)


def test_discovery_substitution_uses_one_complete_protocol() -> None:
    payload = json.loads(AUDIT_PATH.read_text(encoding="utf-8"))
    rows = [
        row for row in payload["rows"] if row["benchmark"] == "DiscoveryBench"
    ]

    assert {row["n"] for row in rows} == {239}
    assert {row["protocol"] for row in rows} == {
        "matched frozen 4-proposal/1-round"
    }
    assert {row["model"] for row in rows} == {
        "GPT-4o-mini",
        "Gemini 2.5 Flash Lite",
        "Qwen3-30B-A3B-Instruct",
    }
