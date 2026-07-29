"""Tiny OpenAI-compatible client fallback for benchmark environments.

Some bundled runtimes used for local evaluation do not ship the official
``openai`` package, while several benchmark harnesses import
``from openai import OpenAI`` directly.  This module implements only the small
chat-completions surface those harnesses use, routing to an OpenAI-compatible
HTTP endpoint with the same environment conventions as ``mars.agents.base``.
If the official package is installed outside this repository, production runs
should prefer that environment instead.
"""

from __future__ import annotations

import json
import os
import signal
import threading
import time
from types import SimpleNamespace
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

__version__ = "0.0-local-fallback"


class _ResponseDeadline:
    """Bound the whole HTTP response, including a slowly streamed body."""

    def __init__(self, seconds: float):
        self.seconds = max(float(seconds), 0.0)
        self._enabled = False
        self._previous_handler: Any = None
        self._previous_timer: tuple[float, float] = (0.0, 0.0)

    @staticmethod
    def _raise_timeout(_signum: int, _frame: Any) -> None:
        raise TimeoutError("OpenAI-compatible response exceeded wall-clock deadline")

    def __enter__(self) -> "_ResponseDeadline":
        self._enabled = (
            self.seconds > 0
            and threading.current_thread() is threading.main_thread()
            and hasattr(signal, "SIGALRM")
            and hasattr(signal, "setitimer")
        )
        if self._enabled:
            self._previous_handler = signal.getsignal(signal.SIGALRM)
            self._previous_timer = signal.getitimer(signal.ITIMER_REAL)
            signal.signal(signal.SIGALRM, self._raise_timeout)
            signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        if not self._enabled:
            return
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, self._previous_handler)
        if self._previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL, *self._previous_timer)


class _Completions:
    def __init__(self, client: "OpenAI"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        payload = dict(kwargs)
        url = self._client.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._client.api_key}",
            "Content-Type": "application/json",
        }
        if "openrouter.ai" in self._client.base_url:
            headers.setdefault("HTTP-Referer", "https://local.mars")
            headers.setdefault("X-Title", "MARS Local Benchmark Runner")
        body = json.dumps(payload).encode("utf-8")
        req = Request(url, data=body, headers=headers, method="POST")
        max_retries = _bounded_int_env("OPENAI_MAX_RETRIES", default=3, lower=0, upper=5)
        retry_base_s = _bounded_float_env(
            "OPENAI_RETRY_BASE_S",
            default=0.5,
            lower=0.0,
            upper=10.0,
        )
        last_error = "empty completion"

        for attempt in range(max_retries + 1):
            try:
                with _ResponseDeadline(self._client.timeout):
                    with urlopen(req, timeout=self._client.timeout) as resp:
                        raw = resp.read().decode("utf-8")
                data = json.loads(raw)
            except HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_error = f"HTTP {exc.code}: {detail}"
                if exc.code not in {429, 500, 502, 503, 504} or attempt >= max_retries:
                    raise RuntimeError(f"OpenAI-compatible {last_error}") from exc
            except URLError as exc:
                last_error = f"request failed: {exc}"
                if attempt >= max_retries:
                    raise RuntimeError(f"OpenAI-compatible {last_error}") from exc
            except TimeoutError as exc:
                # The wall-clock deadline is already the outer safety bound.
                raise RuntimeError(str(exc)) from exc
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                last_error = f"invalid JSON response: {exc}"
                if attempt >= max_retries:
                    raise RuntimeError(f"OpenAI-compatible {last_error}") from exc
            else:
                if _has_usable_completion(data):
                    return _to_namespace(data)
                last_error = "response contained neither message content nor tool calls"
                if attempt >= max_retries:
                    break

            if retry_base_s > 0:
                time.sleep(retry_base_s * (2**attempt))

        raise RuntimeError(
            "OpenAI-compatible completion failed after "
            f"{max_retries + 1} attempts: {last_error}"
        )


class _Chat:
    def __init__(self, client: "OpenAI"):
        self.completions = _Completions(client)


class OpenAI:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        **_: Any,
    ):
        key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY", "")
        url = base_url or os.environ.get("OPENAI_BASE_URL")
        if not url and key.startswith("sk-or-"):
            url = "https://openrouter.ai/api/v1"
        self.api_key = key
        self.base_url = (url or "https://api.openai.com/v1").rstrip("/")
        self.timeout = float(timeout or os.environ.get("OPENAI_TIMEOUT_S", "90"))
        self.chat = _Chat(self)


def _to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_to_namespace(v) for v in value]
    return value


def _has_usable_completion(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return False
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return True
    return bool(message.get("tool_calls"))


def _bounded_int_env(
    name: str,
    *,
    default: int,
    lower: int,
    upper: int,
) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    return min(max(value, lower), upper)


def _bounded_float_env(
    name: str,
    *,
    default: float,
    lower: float,
    upper: float,
) -> float:
    try:
        value = float(os.environ.get(name, str(default)))
    except ValueError:
        value = default
    return min(max(value, lower), upper)
