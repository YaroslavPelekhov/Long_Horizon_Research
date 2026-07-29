from __future__ import annotations

import json
import tempfile
from pathlib import Path

from mars.analysis.build_submission_packages import (
    ROOT,
    _anonymize_local_paths,
    _secret_scan,
)


def test_anonymous_packager_removes_machine_local_paths() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        artifact = directory / "summary.json"
        artifact.write_text(
            json.dumps(
                {
                    "run_log": str(ROOT / "lmw" / "run.jsonl"),
                    "home_cache": str(Path.home() / ".cache" / "model"),
                    "score": 49.38,
                }
            ),
            encoding="utf-8",
        )

        _anonymize_local_paths(directory)
        _secret_scan(directory)

        payload = json.loads(artifact.read_text(encoding="utf-8"))
        assert payload["run_log"] == "lmw/run.jsonl"
        assert payload["home_cache"] == "$HOME/.cache/model"
        assert payload["score"] == 49.38
