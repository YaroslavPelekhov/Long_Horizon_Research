import time
import unittest
from unittest.mock import patch

from openai import OpenAI


class _SlowResponse:
    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        return None

    def read(self):
        time.sleep(0.5)
        return b'{"choices":[{"message":{"content":"late"}}]}'


class _JSONResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        return None

    def read(self):
        return self.payload


class OpenAIFallbackTimeoutTest(unittest.TestCase):
    def test_wall_clock_deadline_interrupts_slow_body(self):
        client = OpenAI(
            api_key="test",
            base_url="https://example.invalid/v1",
            timeout=0.05,
        )
        started = time.monotonic()
        with patch("openai.urlopen", return_value=_SlowResponse()):
            with self.assertRaisesRegex(RuntimeError, "wall-clock deadline"):
                client.chat.completions.create(
                    model="test",
                    messages=[{"role": "user", "content": "hello"}],
                )
        self.assertLess(time.monotonic() - started, 0.3)

    def test_empty_content_is_retried(self):
        client = OpenAI(
            api_key="test",
            base_url="https://example.invalid/v1",
            timeout=1,
        )
        empty = _JSONResponse(b'{"choices":[{"message":{"content":null}}]}')
        valid = _JSONResponse(b'{"choices":[{"message":{"content":"judge answer"}}]}')
        with patch.dict(
            "os.environ",
            {"OPENAI_MAX_RETRIES": "2", "OPENAI_RETRY_BASE_S": "0"},
        ):
            with patch("openai.urlopen", side_effect=[empty, valid]) as mocked:
                result = client.chat.completions.create(
                    model="test",
                    messages=[{"role": "user", "content": "hello"}],
                )
        self.assertEqual(result.choices[0].message.content, "judge answer")
        self.assertEqual(mocked.call_count, 2)

    def test_tool_call_without_content_is_accepted(self):
        client = OpenAI(
            api_key="test",
            base_url="https://example.invalid/v1",
            timeout=1,
        )
        response = _JSONResponse(
            b'{"choices":[{"message":{"content":null,"tool_calls":[{"id":"call_1"}]}}]}'
        )
        with patch.dict(
            "os.environ",
            {"OPENAI_MAX_RETRIES": "2", "OPENAI_RETRY_BASE_S": "0"},
        ):
            with patch("openai.urlopen", return_value=response) as mocked:
                result = client.chat.completions.create(
                    model="test",
                    messages=[{"role": "user", "content": "hello"}],
                )
        self.assertEqual(result.choices[0].message.tool_calls[0].id, "call_1")
        self.assertEqual(mocked.call_count, 1)


if __name__ == "__main__":
    unittest.main()
