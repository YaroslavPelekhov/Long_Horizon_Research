"""
Shared LLM-call helper for MARS agents.

Auto-detects OpenRouter from sk-or- key prefix, retries without
response_format if the first attempt returns empty content (OpenRouter
sometimes rejects structured-output mode silently). Same pattern we proved
out in ols/inner_agents/llm_inner.py.
"""

from __future__ import annotations

import json
import os
from typing import Optional


def make_openai_client():
    """Return an OpenAI client routed to OpenRouter when key starts with sk-or-,
    or honor explicit OPENAI_BASE_URL. Falls back to vanilla OpenAI client."""
    from openai import OpenAI
    key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if not base_url and key.startswith("sk-or-"):
        base_url = "https://openrouter.ai/api/v1"
    if base_url:
        return OpenAI(base_url=base_url, api_key=key)
    return OpenAI()


def call_llm(
    client,
    model: str,
    system: str,
    user: str,
    *,
    max_tokens: int = 700,
    temperature: float = 0.4,
) -> str:
    """One LLM call with json_object → fallback to plain. Returns content
    string or '' on total failure."""
    for use_rf in (True, False):
        try:
            kw = dict(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            if use_rf:
                kw["response_format"] = {"type": "json_object"}
            r = client.chat.completions.create(**kw)
            content = (r.choices[0].message.content or "").strip()
            if content:
                return content
        except Exception:
            continue
    return ""


def parse_json_strict(raw: str) -> Optional[dict]:
    """Best-effort JSON parse: strip fences, find outer braces, parse.
    Returns None on failure."""
    if not raw:
        return None
    s = raw.strip()
    if s.startswith("```"):
        s = s.strip("` \n")
        if s.startswith("json"):
            s = s[4:].lstrip()
    try:
        return json.loads(s)
    except Exception:
        pass
    i, j = s.find("{"), s.rfind("}")
    if 0 <= i < j:
        try:
            return json.loads(s[i:j + 1])
        except Exception:
            return None
    return None
