from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mars.analysis.build_submission_manifest import _discovery, _newton
from mars.induction.universal_cpi import UniversalCPI
from mars.runners.run_db_official_eval import (
    load_official_real_tasks,
    load_official_train_tasks,
)
from mars.runners.run_language_growth_experiment import (
    _language_env,
    _stable_group_split,
)


ROOT = Path(__file__).resolve().parents[1]


class MainExperimentProtocolTests(unittest.TestCase):
    def test_released_discovery_splits_and_group_disjointness(self):
        repo = ROOT / "discoverybench_repo"
        if not repo.is_dir():
            self.skipTest("official DiscoveryBench repository is not bundled")
        train = load_official_train_tasks(
            repo, max_tasks=None, datasets=None, task_keys=None
        )
        test = load_official_real_tasks(
            repo, max_tasks=None, datasets=None, task_keys=None
        )
        induction, promotion = _stable_group_split(
            train, seed=2027, induction_fraction=0.65
        )
        induction_groups = {
            (task.dataset_name, task.metadata_id) for task in induction
        }
        promotion_groups = {
            (task.dataset_name, task.metadata_id) for task in promotion
        }
        self.assertEqual(len(train), 25)
        self.assertEqual(len(test), 239)
        self.assertTrue(induction)
        self.assertTrue(promotion)
        self.assertFalse(induction_groups & promotion_groups)
        self.assertTrue(all(task.task.gold_hypothesis for task in train))

    def test_frozen_eval_loads_but_cannot_mutate_language(self):
        with tempfile.TemporaryDirectory() as td:
            env = _language_env(Path(td), "gated_growth", "frozen_eval")
        self.assertEqual(env["MARS_LOAD_TRUSTED_MODULES"], "1")
        self.assertEqual(env["MARS_LOAD_TRUSTED_LAYERS"], "1")
        self.assertEqual(env["MARS_SELF_WRITE_MODULES"], "0")
        self.assertEqual(env["MARS_SELF_WRITE_LAYERS"], "0")
        self.assertEqual(env["MARS_PROMOTE_SELF_MODULES"], "0")
        self.assertEqual(env["MARS_PROMOTE_SELF_LAYERS"], "0")

    def test_promotion_probe_loads_candidates_but_does_not_propose_new_ones(self):
        with tempfile.TemporaryDirectory() as td:
            env = _language_env(Path(td), "gated_growth", "promotion_probe")
        self.assertEqual(env["MARS_LOAD_CANDIDATE_MODULES"], "1")
        self.assertEqual(env["MARS_LOAD_CANDIDATE_LAYERS"], "1")
        self.assertEqual(env["MARS_PROPOSE_SELF_LAYERS"], "0")
        self.assertEqual(env["MARS_PROMOTION_PROBE"], "1")
        self.assertEqual(env["MARS_SELF_MODULE_MIN_PROBE_KEYS"], "2")

    def test_split_read_write_flags_preserve_legacy_defaults(self):
        engine = UniversalCPI(model="unused")
        with patch.dict(
            "os.environ",
            {
                "MARS_SELF_WRITE_MODULES": "1",
                "MARS_SELF_WRITE_LAYERS": "1",
                "MARS_LOAD_TRUSTED_MODULES": "1",
                "MARS_LOAD_TRUSTED_LAYERS": "1",
                "MARS_PROMOTE_SELF_MODULES": "0",
                "MARS_PROMOTE_SELF_LAYERS": "0",
            },
            clear=False,
        ):
            self.assertTrue(engine._trusted_self_modules_enabled())
            self.assertTrue(engine._trusted_self_layers_enabled())
            self.assertFalse(engine._promote_self_modules_enabled())
            self.assertFalse(engine._promote_self_layers_enabled())

    def test_submission_manifest_rejects_partial_or_rejudged_results(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            discovery_path = root / "discovery.json"
            discovery_path.write_text(
                json.dumps(
                    {
                        "n_tasks": 30,
                        "data_split": "test",
                        "HMS_mean_100": 25.0,
                        "HMS_mean_consistency_100": 27.0,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                _discovery(discovery_path)

            newton_path = root / "newton.json"
            newton_path.write_text(
                json.dumps(
                    {
                        "selective_rejudge": True,
                        "n_per_repetition": 324,
                        "n_repetitions": 4,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                _newton(newton_path)


if __name__ == "__main__":
    unittest.main()
