from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]


def test_main_pdf_has_seven_content_pages_and_two_reference_pages() -> None:
    reader = PdfReader(ROOT / "residual_guided_hli_submission.pdf")
    assert len(reader.pages) == 9

    page_seven = reader.pages[6].extract_text()
    page_eight = reader.pages[7].extract_text()
    assert "Conclusion" in page_seven
    assert "References" not in page_seven
    assert "References" in page_eight


def test_supplement_pdf_has_complete_model_substitution_audit() -> None:
    reader = PdfReader(ROOT / "residual_guided_hli_supplement.pdf")
    assert len(reader.pages) == 11
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    compact = "".join(text.split())
    assert "DiscoveryBenchsubstitution." in compact
    assert "UltraHorizonsubstitution." in compact
    assert "Machine-checkedintegrityconditions" in compact


def test_submission_sources_have_no_unfinished_markers() -> None:
    for name in (
        "residual_guided_hli_submission.tex",
        "residual_guided_hli_supplement.tex",
    ):
        source = (ROOT / name).read_text(encoding="utf-8")
        assert not re.search(
            r"\b(?:TODO|TBD|PLACEHOLDER|working draft|reserved table)\b",
            source,
            flags=re.IGNORECASE,
        )


def test_reproducibility_checklist_has_no_unanswered_questions() -> None:
    source = (ROOT / "ReproducibilityChecklist.tex").read_text(encoding="utf-8")
    question_region = source.split("% The questions start here", maxsplit=1)[1]
    assert "Type your response here" not in question_region
