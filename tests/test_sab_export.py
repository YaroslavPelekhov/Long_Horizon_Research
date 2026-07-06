from mars.runners.run_sab_official_export import (
    OfficialExportSABAdapter,
    _best_submitted_code,
)


def test_sab_export_keeps_last_non_empty_submission():
    adapter = object.__new__(OfficialExportSABAdapter)
    adapter._submitted_program = ""
    adapter._submissions = [
        {"args": {"code": "print('valid')"}},
        {"args": {"code": ""}},
    ]

    assert _best_submitted_code(adapter) == "print('valid')\n"
