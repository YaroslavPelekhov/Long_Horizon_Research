from mars.runners.run_sab_amplification import _import_roots, _normalize_code
from mars.skills.contract_baselines import source_target_tokens


def test_normalize_code_decodes_json_escaped_newlines():
    code = "import os\\nprint('ok')\\n"

    normalized = _normalize_code(code)

    assert normalized == "import os\nprint('ok')"


def test_import_roots_extracts_top_level_dependencies():
    src = "import pandas as pd\nfrom sklearn.metrics import roc_auc_score\nfrom .local import x\n"

    assert _import_roots(src) == {"pandas", "sklearn"}


def test_source_target_tokens_from_output_contract_text():
    task = {
        "task_inst": "Train mapping from Sub 01 to another Subject 3.",
        "output_fname": "pred_results/linear_sub01tosub03_pred.npy",
    }

    assert source_target_tokens(task) == ("sub01", "sub03")
