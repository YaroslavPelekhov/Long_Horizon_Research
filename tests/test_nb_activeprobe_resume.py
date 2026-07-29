import json
from pathlib import Path

from mars.runners.run_nb_activeprobe import _load_resume_checkpoint


def test_partial_checkpoint_has_unique_task_keys(tmp_path: Path) -> None:
    checkpoint = tmp_path / "summary.json"
    checkpoint.write_text(
        json.dumps(
            {
                "results": {
                    "m0/easy/v0/vanilla_equation": {
                        "status": "ANSWER",
                        "SA": 1.0,
                    },
                    "m0/easy/v0/simple_system": {
                        "status": "ABSTAIN",
                        "SA": 0.0,
                    },
                },
                "rows": [{"stale": True}],
            }
        ),
        encoding="utf-8",
    )

    results = _load_resume_checkpoint(checkpoint)
    rows = list(results.values())

    assert list(results) == [
        "m0/easy/v0/vanilla_equation",
        "m0/easy/v0/simple_system",
    ]
    assert len(rows) == 2
    assert sum(row["status"] == "ANSWER" for row in rows) == 1
