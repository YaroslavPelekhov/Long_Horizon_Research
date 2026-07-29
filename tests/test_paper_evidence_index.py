from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from mars.analysis.build_paper_evidence_index import EVIDENCE


ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "paper_assets" / "evidence"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_paper_evidence_index_is_complete_and_current() -> None:
    csv_path = INDEX_DIR / "evidence_index.csv"
    json_path = INDEX_DIR / "evidence_index.json"
    csv_rows = list(csv.DictReader(csv_path.open(encoding="utf-8", newline="")))
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    json_rows = payload["entries"]

    expected_ids = {entry[0] for entry in EVIDENCE}
    assert {row["claim_id"] for row in csv_rows} == expected_ids
    assert {row["claim_id"] for row in json_rows} == expected_ids
    assert payload["all_sources_hashed"] is True

    missing = [row["artifact"] for row in json_rows if not (ROOT / row["artifact"]).is_file()]
    if missing:
        pytest.skip("full evaluator artifact bundle is not mounted")

    for row in json_rows:
        source = ROOT / row["artifact"]
        assert row["bytes"] == source.stat().st_size
        assert row["sha256"] == _sha256(source)


def test_headline_artifacts_have_full_cardinality() -> None:
    payload = json.loads(
        (INDEX_DIR / "evidence_index.json").read_text(encoding="utf-8")
    )
    rows = {row["claim_id"]: row for row in payload["entries"]}

    assert rows["headline.discovery.summary"]["records"] == 239
    assert rows["headline.discovery.rows"]["records"] == 239
    assert rows["headline.newton.summary"]["records"] == 324
    assert rows["headline.newton.rows"]["records"] == 324
    assert rows["headline.ultrahorizon.summary"]["records"] == 96
    assert rows["headline.ultrahorizon.rows"]["records"] == 96
