import os
import time
import unittest
from unittest.mock import patch

try:
    from newtonbench_repo.utils.code_executor_base import CodeExecutorBase
except ModuleNotFoundError:
    CodeExecutorBase = None


@unittest.skipIf(CodeExecutorBase is None, "official NewtonBench repository is not bundled")
class NewtonCodeExecutorTimeoutTests(unittest.TestCase):
    def test_infinite_python_action_is_interrupted(self):
        executor = object.__new__(CodeExecutorBase)
        started = time.monotonic()
        with patch.dict(
            os.environ, {"NEWTONBENCH_CODE_TIMEOUT_S": "1"}, clear=False
        ):
            result = executor.execute_python_code("while True:\n    pass")
        self.assertFalse(result["success"])
        self.assertEqual(result["error_type"], "execution_timeout")
        self.assertLess(time.monotonic() - started, 2.0)

    def test_matplotlib_show_is_headless_and_non_blocking(self):
        executor = object.__new__(CodeExecutorBase)
        started = time.monotonic()
        result = executor.execute_python_code(
            "import matplotlib.pyplot as plt\n"
            "plt.plot([0, 1], [0, 1])\n"
            "plt.show()\n"
            "print('done')"
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["stdout"], "done")
        self.assertLess(time.monotonic() - started, 2.0)


if __name__ == "__main__":
    unittest.main()
