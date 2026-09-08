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
    "Rewrite in caveman/telegraphic English. Preserve all factual information. "
    "Remove articles, unnecessary pronouns, auxiliary/linking verbs, optional "
    "prepositions, infinitive “to,” and other grammatical filler when meaning "
    "remains clear. Prefer noun phrases and compact subject–verb–object "
    "structures. Do not summarize or omit facts. Preserve names, numbers, "
    "dates, negation, comparisons, causality, and relationships. Grammatical "
    "correctness is not required.\n"
    "\n"
    "Generalize with rules like these:\n"
    "Remove articles: a, an, the.\n"
    "the final game in the series → final game in series\n"
    "Remove pronouns when referent obvious: it, they, he, she, that.\n"
    "said it was intended → said intended\n"
    "Remove many helper/linking verbs: is, are, was, were, has been.\n"
    "game was a commercial success → game commercial success\n"
    "Keep main verbs, but simplify surrounding grammar.\n"
    "was intended to be → intended be\n"
    "was ported to Windows → ported to Windows\n"
    "Remove infinitive “to” when meaning remains clear.\n"
    "intended to be final → intended be final\n"
    "attempts to challenge → attempts challenge\n"
    "Remove possessive/function words where obvious.\n"
    "the design of John's upgrades → Bob upgrade design\n"
    "Prefer noun stacking.\n"
    "the presentation of the story → story presentation\n"
    "parts for the two devices → two device parts\n"
    "Replace verbose constructions with shorter equivalents.\n"
    "in order to → to\n"
    "is able to → can\n"
    "a series of → omit or use multiple\n"
    "at the point of impact → at impact\n"
    "Remove repeated subjects.\n"
    "John focuses on distance combat... Bob uses... → John: distance combat... "
    "Bob: close combat...\n"
    "Keep names, numbers, dates, places, actions, relationships, negation, and "
    "causal information. These carry most factual meaning.\n"
    "Do not remove words when doing so creates ambiguity.\n"
    "Bob does not use armor must retain not.\n"
    "John defeated Bob cannot become John Bob defeated."
)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def build_user_prompt(
    query: str,
    title: str | None,
    url: str,
    text: str,
    mode: str = "summary",
) -> str:
    clipped = (text or "")[: config.SUMMARY_INPUT_MAX_CHARS]
    if mode == "original-caveman":
        # Full-rewrite mode: no meta header — the model otherwise echoes
        # the "Search query / Page title / Page URL" labels into the output.
        return (
            "Rewrite the following page content in caveman/telegraphic style. "
            "Preserve every fact. Omit nothing. "
            "Output only the rewritten content, no preamble.\n\n"
            f"{clipped}"
        )
    if mode == "summary-caveman":
        # Identical task to "summary" — only the system prompt differs
        # (telegraphic style). No selection/compression of its own, so no
        # information is lost versus the plain summary.
        task = (
            "Task: Summarize the page content above in detail. "
            "Focus on relevance to the search query where applicable."
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
    query: str,
    title: str | None,
    url: str,
    text: str,
    mode: str | None = None,
) -> str:
    """Return summary, or '' on failure (caller falls back).

    mode=None follows the MODE config ("summary", "summary-caveman",
    "original", "original-caveman"). The test UI forces explicit modes so
    every pane follows the same rules as the Mode setting — no frozen copies.
    """
    if not (text or "").strip():
        return ""
    use_mode = config.MODE if mode is None else mode
    use_caveman = use_mode in ("summary-caveman", "original-caveman")
    if use_caveman:
        system_prompt = CAVEMAN_SYSTEM_PROMPT
    else:
        system_prompt = SYSTEM_PROMPT
    endpoint = config.LLM_BASE_URL.rstrip("/") + "/chat/completions"
    body: dict = {
        "model": config.LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": build_user_prompt(query, title, url, text, mode=use_mode),
            },
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
