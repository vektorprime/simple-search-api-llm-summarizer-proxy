"""Minimal Exa Search API client.

Uses POST /search with contents.text + contents.highlights so we have
full-ish text for the local summarizer, with highlights as fallback.
"""
from __future__ import annotations

import logging

import httpx

from app import config

log = logging.getLogger(__name__)


async def exa_search(query: str, num_results: int) -> list[dict]:
    """Call Exa /search, return raw result dicts. Returns [] on any error."""
    if not config.EXA_API_KEY:
        log.error("EXA_API_KEY is not set")
        return []

    # 0/negative = unlimited: omit maxCharacters so Exa returns its default text.
    text_option: bool | dict = (
        True if config.EXA_TEXT_MAX_CHARS <= 0 else {"maxCharacters": config.EXA_TEXT_MAX_CHARS}
    )
    payload = {
        "query": query,
        "numResults": max(1, min(num_results, 20)),
        "type": "auto",
        "contents": {
            "text": text_option,
            "highlights": True,
        },
    }
    headers = {
        "x-api-key": config.EXA_API_KEY,
        "Content-Type": "application/json",
    }
    url = config.EXA_BASE_URL.rstrip("/") + "/search"
    try:
        async with httpx.AsyncClient(timeout=config.EXA_TIMEOUT_SEC) as client:
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        log.warning("Exa search failed: %s", e)
        return []

    results = data.get("results", []) if isinstance(data, dict) else []
    return results if isinstance(results, list) else []
