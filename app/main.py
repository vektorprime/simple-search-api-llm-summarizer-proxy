"""OpenWebUI-compatible external search proxy.

OpenWebUI contract (engine=external):
  POST /search  {"query": str, "count": int}
  -> [{"link": str, "title": str|None, "snippet": str|None}]
  Auth: any (or no) Bearer key accepted on /search by design.

Flow per request:
  OpenWebUI -> Exa /search (text+highlights) -> LLM backend summarizes each
  -> snippet = summary or highlights/text fallback.
"""
from __future__ import annotations

import asyncio
import logging
import pathlib
import re
import secrets

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from app import config
from app.exa import exa_search
from app.models import SearchRequest, SearchResult
from app.summarizer import summarize_one

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("search-proxy")

app = FastAPI(title="SSALMP", version="0.1.0")
_basic = HTTPBasic()


def _check_admin(creds: HTTPBasicCredentials = Depends(_basic)) -> None:
    ok_user = secrets.compare_digest(creds.username, config.ADMIN_USER)
    ok_pass = secrets.compare_digest(creds.password, config.ADMIN_PASS)
    if not (ok_user and ok_pass):
        raise HTTPException(
            status_code=401,
            detail="Admin login required",
            headers={"WWW-Authenticate": "Basic"},
        )


def _check_auth(authorization: str | None) -> None:
    # Open endpoint by design: accept any (or no) API key from OpenWebUI.
    # PROXY_API_KEY is kept for display only and is NOT enforced.
    return


def _fallback_snippet(item: dict) -> tuple[str, str]:
    """Return (snippet, source). Source is 'highlights', 'text' or 'empty'."""
    highlights = item.get("highlights") or []
    if isinstance(highlights, list) and highlights:
        joined = "\n".join(h for h in highlights if isinstance(h, str) and h.strip())
        if joined.strip():
            return joined[:4000], "highlights"
    text = item.get("text") or ""
    if isinstance(text, str) and text.strip():
        return text[:2000], "text"
    return "", "empty"


_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\((https?://[^)\s]+)\)")


def _image_urls(item: dict) -> list[str]:
    """Representative image + any markdown-embedded image URLs, deduped."""
    urls: list[str] = []
    img = item.get("image")
    if isinstance(img, str) and img.startswith("http"):
        urls.append(img)
    text = item.get("text") or ""
    if isinstance(text, str):
        for m in _MD_IMAGE_RE.findall(text):
            if m not in urls:
                urls.append(m)
    return urls


def _with_images(snippet: str, image_urls: list[str]) -> str:
    if not image_urls:
        return snippet
    block = "\n\nImages on page:\n" + "\n".join(f"- {u}" for u in image_urls)
    return (snippet + block) if snippet else block.strip()


async def _summarize_item(query: str, item: dict, mode: str | None = None) -> tuple[SearchResult, dict]:
    """Return (openwebui_result, debug_info with raw search fields + summary source)."""
    use_mode = config.MODE if mode is None else mode
    url = item.get("url") or item.get("link") or ""
    title = item.get("title")
    text = item.get("text") or ""
    highlights = item.get("highlights") if isinstance(item.get("highlights"), list) else []
    image_urls = _image_urls(item) if config.RETURN_IMAGE_URLS else []
    snippet, source = "", "empty"
    slot_id: int | None = None
    if use_mode == "original":
        if text.strip():
            snippet, source = text, "original"
    elif text.strip():
        # Global single-flight: only MAX_CONCURRENT_SUMMARIES summaries run
        # against the backend at once (default 1); everything else queues here.
        async with _global_summary_sem():
            slot_id = _rotated_slot_id()
            snippet = await summarize_one(
                query, title, url, text, mode=use_mode, slot_id=slot_id
            )
        if snippet:
            source = "llm"
    else:
        slot_id = None
    if not snippet:
        snippet, source = _fallback_snippet(item)
    snippet = _with_images(snippet, image_urls)
    debug = {
        "link": url,
        "title": title,
        "snippet": snippet,
        "snippet_source": source,
        "slot_id": slot_id,
        "exa_text": text,
        "exa_highlights": highlights,
        "image_urls": image_urls,
    }
    return SearchResult(link=url, title=title, snippet=snippet), debug


# llama.cpp slot rotation: each dispatched job takes the next id in
# 0..SLOT_COUNT-1, so concurrent jobs land on different slots instead of
# piling onto one. A job hitting a busy slot waits its turn server-side.
_slot_counter = 0


def _rotated_slot_id() -> int | None:
    """Next pool slot, or None when slot management is off / N/A."""
    global _slot_counter
    if config.LLM_PROVIDER != "llamacpp" or not config.LLAMACPP_USE_SLOTS:
        return None
    n = max(1, config.LLAMACPP_SLOT_COUNT)
    slot = _slot_counter % n
    _slot_counter += 1
    return slot


# Global semaphore shared by ALL requests (not per-request), so the LLM
# backend never sees more than MAX_CONCURRENT_SUMMARIES concurrent prompts.
# Rebuilt when the running loop or the configured size changes
# (size is tunable at runtime via /admin).
_global_sem: asyncio.Semaphore | None = None
_global_sem_key: tuple[int, int] | None = None


def _global_summary_sem() -> asyncio.Semaphore:
    global _global_sem, _global_sem_key
    size = max(1, config.MAX_CONCURRENT_SUMMARIES)
    key = (id(asyncio.get_running_loop()), size)
    if _global_sem is None or _global_sem_key != key:
        _global_sem = asyncio.Semaphore(size)
        _global_sem_key = key
    return _global_sem


@app.get("/healthz")
async def healthz() -> dict:
    return {
        "status": "ok",
        "llm_model": config.LLM_MODEL,
        "llm_base_url": config.LLM_BASE_URL,
        "exa_configured": bool(config.EXA_API_KEY),
    }


@app.get("/")
async def root() -> dict:
    return {
        "service": "ssalmp",
        "name": "Simple Search API LLM Summarizer Proxy",
        "usage": "POST /search with {query, count}; GET /healthz; GET /admin",
        "openwebui_engine": "external",
    }


@app.get("/config")
async def get_config(_: None = Depends(_check_admin)) -> JSONResponse:
    return JSONResponse(content=config.public_config())


@app.post("/config")
async def post_config(req: Request, _: None = Depends(_check_admin)) -> JSONResponse:
    try:
        updates = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    if not isinstance(updates, dict):
        raise HTTPException(status_code=400, detail="Body must be a JSON object")
    allowed = {k: v for k, v in updates.items() if k in config.EDITABLE_FIELDS}
    applied = config.update_config(allowed)
    return JSONResponse(content={"applied": applied, "config": config.public_config()})


@app.get("/admin", response_class=HTMLResponse)
async def admin_page(_: None = Depends(_check_admin)) -> str:
    return ADMIN_HTML


@app.get("/llm/models")
async def llm_models(_: None = Depends(_check_admin)) -> JSONResponse:
    """Admin-only: autodetect model ids from the LLM backend's /v1/models.

    Works with llama.cpp, vLLM and SGLang (all OpenAI-compatible).
    Returns {"models": [...], "base_url": ...}; 502 if the backend is down.
    """
    url = config.LLM_BASE_URL.rstrip("/") + "/models"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                url, headers={"Authorization": f"Bearer {config.LLM_API_KEY}"}
            )
            resp.raise_for_status()
            data = resp.json()
        ids = [
            m.get("id")
            for m in (data.get("data") or [])
            if isinstance(m, dict) and m.get("id")
        ]
        return JSONResponse(content={"models": ids, "base_url": config.LLM_BASE_URL})
    except Exception as e:
        log.warning("llm model autodetect failed: %s", e)
        raise HTTPException(status_code=502, detail=f"Could not list models: {e}")


@app.get("/llm/slots")
async def llm_slots(_: None = Depends(_check_admin)) -> JSONResponse:
    """Admin-only: list llama.cpp slots (id + busy state) for the Detect button.

    llama.cpp serves this natively at GET /slots (no /v1 prefix), so the
    trailing /v1 is stripped from the base URL. Other backends 404 → 502.
    Returns {"count": N, "slots": [{"id": 0, "busy": false}, ...]}.
    Slot ids are 0-based.
    """
    base = config.LLM_BASE_URL.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    url = base + "/slots"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                url, headers={"Authorization": f"Bearer {config.LLM_API_KEY}"}
            )
            resp.raise_for_status()
            data = resp.json()
        if not isinstance(data, list):
            raise ValueError("unexpected /slots shape")
        slots = [
            {"id": s.get("id"), "busy": bool(s.get("is_processing"))}
            for s in data
            if isinstance(s, dict) and isinstance(s.get("id"), int)
        ]
        return JSONResponse(content={"count": len(slots), "slots": slots, "slots_url": url})
    except Exception as e:
        log.warning("llm slot detect failed: %s", e)
        raise HTTPException(status_code=502, detail=f"Could not list slots: {e}")


@app.post("/search")
async def search(
    req: SearchRequest, authorization: str | None = Header(default=None)
) -> JSONResponse:
    _check_auth(authorization)
    if not req.query.strip():
        return JSONResponse(content=[])
    try:
        items = await asyncio.wait_for(
            exa_search(req.query, req.count),
            timeout=config.REQUEST_TIMEOUT_SEC,
        )
    except Exception as e:
        log.warning("search pipeline failed: %s", e)
        return JSONResponse(content=[])

    if not items:
        return JSONResponse(content=[])

    try:
        pairs = await asyncio.wait_for(
            asyncio.gather(*(_summarize_item(req.query, it) for it in items[: req.count])),
            timeout=config.REQUEST_TIMEOUT_SEC,
        )
    except Exception as e:
        log.warning("summarization pipeline failed: %s", e)
        return JSONResponse(content=[])

    # drop items with no URL; OpenWebUI tolerates fewer than `count`
    payload = [r.model_dump() for r, _ in pairs if r.link]
    return JSONResponse(content=payload)


@app.post("/debug/search")
async def debug_search(req: SearchRequest, _: None = Depends(_check_admin)) -> JSONResponse:
    """Admin-only: full pipeline trace per result.

    Returns {"mode": ..., "results": [{link, title, snippet, snippet_source,
    exa_text, exa_highlights, image_urls, comparison_summary}]} so the admin
    UI can show the raw original next to exactly what OpenWebUI receives.
    `comparison_summary` is a normal-style summary computed with the same
    rules as the Summary mode (null when already in summary/original modes
    with nothing to compare). There are no frozen prompt copies anywhere:
    every pane is produced from the live MODE rules.
    """
    if not req.query.strip():
        return JSONResponse(content={"results": []})
    try:
        items = await asyncio.wait_for(
            exa_search(req.query, req.count),
            timeout=config.REQUEST_TIMEOUT_SEC,
        )
    except Exception as e:
        log.warning("debug search pipeline failed: %s", e)
        return JSONResponse(content={"results": [], "error": str(e)})
    if not items:
        return JSONResponse(content={"results": []})
    try:
        pairs = await asyncio.wait_for(
            asyncio.gather(*(_summarize_item(req.query, it) for it in items[: req.count])),
            timeout=config.REQUEST_TIMEOUT_SEC,
        )
    except Exception as e:
        log.warning("debug summarization pipeline failed: %s", e)
        return JSONResponse(content={"results": [], "error": str(e)})
    results = [dbg for r, dbg in pairs if r.link]
    if config.MODE == "summary":
        for dbg in results:
            dbg["comparison_summary"] = None
    else:
        # One extra normal-style summary per result so the UI can compare
        # the current mode's output against the Summary mode rules.
        # Reuses the global single-flight semaphore.
        async def _comparison(dbg: dict) -> None:
            text = dbg.get("exa_text") or ""
            if not text.strip():
                dbg["comparison_summary"] = None
                return
            async with _global_summary_sem():
                normal = await summarize_one(
                    req.query, dbg.get("title"), dbg.get("link") or "", text,
                    mode="summary", slot_id=_rotated_slot_id(),
                )
            dbg["comparison_summary"] = normal or None

        try:
            await asyncio.wait_for(
                asyncio.gather(*(_comparison(dbg) for dbg in results)),
                timeout=config.REQUEST_TIMEOUT_SEC,
            )
        except Exception as e:
            log.warning("debug comparison summaries failed: %s", e)
            for dbg in results:
                dbg.setdefault("comparison_summary", None)
    return JSONResponse(content={"mode": config.MODE, "results": results})


def _admin_html() -> str:
    """Serve the admin UI from app/admin.html (same container, no extra deps)."""
    try:
        return pathlib.Path(__file__).with_name("admin.html").read_text(encoding="utf-8")
    except OSError:
        return "<h1>Admin UI missing (admin.html not found)</h1>"


ADMIN_HTML = _admin_html()
