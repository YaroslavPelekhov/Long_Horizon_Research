import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
METADATA = json.loads(
    (ROOT / "paper_assets" / "submission_metadata.json").read_text(encoding="utf-8")
)


def _normalize_tex_prose(text: str) -> str:
    text = re.sub(r"(?<!\\)%.*", "", text)
    text = text.replace("\\%", "%")
    return " ".join(text.split())


def _extract_title(source: str) -> str:
    match = re.search(r"\\title\{([^{}]+)\}", source)
    assert match is not None
    return _normalize_tex_prose(match.group(1))


def _extract_abstract(source: str) -> str:
    match = re.search(
        r"\\begin\{abstract\}(.*?)\\end\{abstract\}",
        source,
        flags=re.DOTALL,
    )
    assert match is not None
    return _normalize_tex_prose(match.group(1))


def test_registered_title_and_abstract_are_frozen() -> None:
    sources = [ROOT / "residual_guided_hli_submission.tex"]
    legacy_fullfit = ROOT / "residual_guided_hli_submission_fullfit.tex"
    if legacy_fullfit.is_file():
        sources.append(legacy_fullfit)
    for path in sources:
        source = path.read_text(encoding="utf-8")
        assert _extract_title(source) == METADATA["title"]
        assert _extract_abstract(source) == METADATA["abstract"]
