"""Summarize page text with the configured OpenAI-compatible LLM backend.

Key backend behaviour to be aware of (verified against llama.cpp
with --reasoning on, 2026-09-08):
- Server runs with --reasoning on, so part of max_tokens is spent on
  reasoning_content. Small max_tokens (e.g. 50) returns empty content
  with finish_reason=length. Always request >=1500 tokens.
- Response shape: choices[0].message = {role, content, reasoning_content?}.
  We use only `content` and strip any leaked <think> blocks.
"""
from __future__ import annotations

import logging
import re

import httpx

from app import config

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a precise web content summarizer. "
    "Summarize the provided page content in detail. "
    "Preserve key facts, names, numbers, and dates. "
    "Return only the summary, no preamble and no extra commentary."
)

CAVEMAN_SYSTEM_PROMPT = (
    "ACTIVE EVERY RESPONSE. No revert after many turns. No filler drift. "
    'Still active if unsure. Off only: "stop caveman" / "normal mode".\n'
    "\n"
    "Rules\n"
    "\n"
    "Drop: articles (a/an/the), filler (just/really/basically/actually/simply), "
    "pleasantries (sure/certainly/of course/happy to), hedging. Fragments OK. "
    'Short synonyms (big not extensive, fix not "implement a solution for"). '
    "Technical terms exact. Code blocks unchanged. Errors quoted exact.\n"
    "\n"
    "Pattern: [thing] [action] [reason]. [next step].\n"
    "\n"
    "Not: \"Sure! I'd be happy to help you with that. The issue you're "
    "experiencing is likely caused by...\" Yes: \"Bug in auth middleware. "
    "Token expiry check use < not <=. Fix:\"\n"
    "\n"
    "No filler/hedging. Keep articles + full sentences. Professional but tight\n"
    "\n"
    'Example — "Why React component re-render?"\n'
    "\n"
    '"Your component re-renders because you create a new object reference each '
    'render. Wrap it in useMemo."\n'
    "\n"
    'Example — "Explain database connection pooling."\n'
    "\n"
    '"Connection pooling reuses open connections instead of creating new ones '
    'per request. Avoids repeated handshake overhead."'
)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def build_user_prompt(
    query: str, title: str | None, url: str, text: str, caveman: bool | None = None
) -> str:
    use_caveman = config.CAVEMAN_STYLE if caveman is None else caveman
    clipped = (text or "")[: config.SUMMARY_INPUT_MAX_CHARS]
    if use_caveman:
        task = (
            "Task: Summarize page content above like caveman. "
            "Very few words. Short sentences. Keep key facts."
        )
    else:
        task = (
            "Task: Summarize the page content above in detail. "
            "Focus on relevance to the search query where applicable."
        )
    return (
        f"Search query: {query}\n"
        f"Page title: {title or 'n/a'}\n"
        f"Page URL: {url}\n\n"
        f"Page content:\n{clipped}\n\n"
        f"{task}"
    )


def clean_summary(text: str | None) -> str:
    if not text:
        return ""
    text = _THINK_RE.sub("", text).strip()
    # also handle unclosed / stray think tags from reasoning-preserve mode
    text = text.replace("<think>", "").replace("</think>", "").strip()
    return text


async def summarize_one(
    query: str, title: str | None, url: str, text: str, caveman: bool | None = None
) -> str:
    """Return summary, or '' on failure (caller falls back).

    caveman=None follows the CAVEMAN_STYLE config; True/False forces the style.
    """
    if not (text or "").strip():
        return ""
    use_caveman = config.CAVEMAN_STYLE if caveman is None else caveman
    endpoint = config.LLM_BASE_URL.rstrip("/") + "/chat/completions"
    body: dict = {
        "model": config.LLM_MODEL,
        "messages": [
            {"role": "system", "content": CAVEMAN_SYSTEM_PROMPT if use_caveman else SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(query, title, url, text, caveman=use_caveman)},
        ],
        # NOTE: sampling params (temp/top-p/top-k) always come from the
        # inference engine defaults. max_tokens is only sent when positive;
        # LLM_MAX_TOKENS <= 0 omits it so the server default applies
        # (llama.cpp default: -1 = unlimited, generate until EOS).
        "stream": False,
    }
    if config.LLM_MAX_TOKENS > 0:
        # Headroom matters: server runs --reasoning on, reasoning tokens
        # come out of this budget; small values return empty content.
        body["max_tokens"] = config.LLM_MAX_TOKENS
    if config.LLM_PROVIDER == "llamacpp" and config.LLAMACPP_USE_SLOTS:
        # Pin the job to a specific llama.cpp slot (id_slot; -1 = any idle).
        body["id_slot"] = int(config.LLAMACPP_SLOT_ID)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.LLM_API_KEY}",
    }
    try:
        async with httpx.AsyncClient(timeout=config.LLM_TIMEOUT_SEC) as client:
            resp = await client.post(endpoint, json=body, headers=headers)
            resp.raise_for_status()
            data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            return ""
        content = choices[0].get("message", {}).get("content")
        return clean_summary(content)
    except Exception as e:
        log.warning("LLM summarize failed for %s: %s", url, e)
        return ""
