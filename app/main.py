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
import collections
import logging
import pathlib
import re
import secrets

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from app import cache as cache_mod
from app import config
from app.exa import exa_search
from app.models import SearchRequest, SearchResult
from app.splitter import split_text
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


def _check_admin_write(request: Request, _: None = Depends(_check_admin)) -> None:
    """Admin auth + JSON-only body for state-changing admin endpoints.

    Browsers resend cached Basic credentials, so a cross-site HTML form could
    otherwise POST here (a text/plain body that happens to parse as JSON).
    Forms cannot send application/json without a CORS preflight, which this
    app never grants — requiring it blocks that CSRF path.
    """
    ctype = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if ctype != "application/json":
        raise HTTPException(status_code=415, detail="Content-Type must be application/json")


if config.ADMIN_USER == "admin" and config.ADMIN_PASS == "admin":
    log.warning("admin login is the default admin/admin — set ADMIN_USER/ADMIN_PASS "
                "in .env; anyone reaching this port can change settings")


def _check_auth(authorization: str | None) -> None:
    # Open endpoint by design: accept any (or no) API key from OpenWebUI.
    # PROXY_API_KEY is kept for display only and is NOT enforced.
    return


def _dedupe_items(items: list[dict]) -> list[dict]:
    """Drop repeat URLs within one search (first occurrence wins).

    Exa occasionally returns the same URL twice in one response; without
    this we'd summarize (and bill LLM calls for) the identical page twice.
    Uses the link cache's URL normalization (host case, trailing slash,
    fragment), so two results never race for the same link-cache key.
    """
    seen: set[str] = set()
    out: list[dict] = []
    for it in items:
        url = cache_mod.normalize_url(it.get("url") or it.get("link") or "")
        if not url or url in seen:
            if url:
                log.info("dropping duplicate search result: %s", url)
            continue
        seen.add(url)
        out.append(it)
    return out


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
    # Defensive: image URLs never mix with original pass-through (config
    # forces RETURN_IMAGE_URLS off in that mode; this guards odd env combos).
    want_images = config.RETURN_IMAGE_URLS and use_mode != "original"
    image_urls = _image_urls(item) if want_images else []
    snippet, source = "", "empty"
    slot_id: int | None = None
    part_summaries: list[dict] | None = None
    passthrough = False
    # True when an LLM call was made and failed (fully, or for some chunked
    # parts). Such results are served but never cached, so a degraded answer
    # is not frozen after the backend recovers. Text-less pages never reach
    # the LLM, so their highlight fallback is deterministic and cacheable.
    llm_failed = False
    if (
        use_mode != "original"
        and text.strip()
        and config.SUMMARY_MIN_CHARS > 0
        and len(text) < config.SUMMARY_MIN_CHARS
    ):
        # Below the trigger size: hand the raw page to the downstream LLM
        # untouched — no LLM call, no rewriting, no image footer.
        snippet, source = text, "passthrough"
        passthrough = True
    elif use_mode == "original":
        if text.strip():
            snippet, source = text, "original"
    elif text.strip():
        parts = (
            split_text(text, config.CHUNK_TARGET_CHARS)
            if config.CHUNKED_SUMMARY and len(text) > config.CHUNK_MIN_CHARS
            else None
        )
        if parts is not None:
            # Chunked: summarize each part, join deterministically. No final
            # LLM call. Dropped parts are logged and simply omitted.
            async def _summarize_part(i: int, heading: str, chunk: str) -> dict:
                async with _global_summary_sem():
                    slot = _rotated_slot_id()
                    out = await summarize_one(
                        query, title, url, chunk, mode=use_mode, slot_id=slot,
                        part=(i + 1, len(parts)), section_heading=heading or None,
                    )
                return {"heading": heading, "summary": out, "slot_id": slot}

            part_summaries = await asyncio.gather(
                *(_summarize_part(i, h, c) for i, (h, c) in enumerate(parts))
            )
            good = [s for s in part_summaries if s["summary"]]
            dropped = len(parts) - len(good)
            if dropped:
                llm_failed = True
                log.warning("chunked summary: dropped %d/%d parts for %s", dropped, len(parts), url)
            snippet = "\n\n".join(
                f"## {p['heading'] or 'Part ' + str(i + 1)}\n\n{p['summary']}".strip()
                for i, p in enumerate(part_summaries) if p["summary"]
            ).strip()
            if snippet:
                source = "llm-parts"
        else:
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
                llm_failed = True
    else:
        slot_id = None
    if not snippet:
        snippet, source = _fallback_snippet(item)
    if not passthrough:
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
        "part_summaries": part_summaries,
        "cacheable": not llm_failed,
    }
    return SearchResult(link=url, title=title, snippet=snippet), debug


def _fallback_result(item: dict, reason: str) -> tuple[SearchResult, dict]:
    """Highlights/text result for an item whose summary timed out or crashed.

    Same shape as _summarize_item's output; never cacheable.
    """
    url = item.get("url") or item.get("link") or ""
    title = item.get("title")
    text = item.get("text") or ""
    want_images = config.RETURN_IMAGE_URLS and config.MODE != "original"
    image_urls = _image_urls(item) if want_images else []
    snippet, source = _fallback_snippet(item)
    snippet = _with_images(snippet, image_urls)
    debug = {
        "link": url,
        "title": title,
        "snippet": snippet,
        "snippet_source": source,
        "slot_id": None,
        "exa_text": text,
        "exa_highlights": item.get("highlights") if isinstance(item.get("highlights"), list) else [],
        "image_urls": image_urls,
        "part_summaries": None,
        "cacheable": False,
        "fallback_reason": reason,
    }
    return SearchResult(link=url, title=title, snippet=snippet), debug


async def _summarize_all(
    query: str, items: list[dict], timeout: float, on_done=None
) -> list[tuple[SearchResult, dict]]:
    """Summarize items concurrently under one deadline, keeping partial work.

    Each finished item is passed to on_done(index, result) as soon as it
    completes (so the link cache fills even if the request later times
    out). At the deadline, unfinished items are cancelled and fall back to
    highlights/text; an item that raises falls back the same way. Results
    are returned in input order — one slow page never empties the search.
    """
    async def _run(i: int, it: dict) -> tuple[SearchResult, dict]:
        res = await _summarize_item(query, it)
        if on_done is not None:
            on_done(i, res)
        return res

    tasks = [asyncio.create_task(_run(i, it)) for i, it in enumerate(items)]
    if not tasks:
        return []
    try:
        _, pending = await asyncio.wait(tasks, timeout=max(0.0, timeout))
        if pending:
            log.warning("summary deadline hit: %d/%d results fell back to highlights/text",
                        len(pending), len(tasks))
    finally:
        # Also covers the client disconnecting mid-search: no orphan tasks
        # keep holding summary slots.
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    out: list[tuple[SearchResult, dict]] = []
    for t, it in zip(tasks, items):
        if t.cancelled():
            out.append(_fallback_result(it, "timeout"))
        elif t.exception() is not None:
            log.warning("summarize failed for %s: %r", it.get("url"), t.exception())
            out.append(_fallback_result(it, "error"))
        else:
            out.append(t.result())
    return out


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


class _SummaryLimiter:
    """FIFO concurrency cap that reads MAX_CONCURRENT_SUMMARIES live.

    Unlike swapping in a fresh asyncio.Semaphore on resize (which let old
    holders and new entrants run side by side, exceeding the cap), one
    counter is kept across resizes: shrinking the limit lets in-flight jobs
    finish and admits no one until the count drops below it; growing it
    admits queued jobs at once (call wake()). All bookkeeping is synchronous,
    so it is safe without a lock on a single event loop.
    """

    def __init__(self) -> None:
        self._active = 0
        self._waiters: collections.deque[asyncio.Future] = collections.deque()

    @staticmethod
    def _limit() -> int:
        return max(1, config.MAX_CONCURRENT_SUMMARIES)

    def wake(self) -> None:
        while self._waiters and self._active < self._limit():
            fut = self._waiters.popleft()
            if not fut.done():
                self._active += 1  # slot handed straight to the waiter
                fut.set_result(None)

    async def __aenter__(self) -> None:
        if not self._waiters and self._active < self._limit():
            self._active += 1
            return
        fut = asyncio.get_running_loop().create_future()
        self._waiters.append(fut)
        try:
            await fut
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled():
                # slot was granted just as we were cancelled: give it back
                self._active -= 1
                self.wake()
            else:
                try:
                    self._waiters.remove(fut)
                except ValueError:
                    pass
            raise

    async def __aexit__(self, *exc) -> None:
        self._active -= 1
        self.wake()


# Global limiter shared by ALL requests (not per-request), so the LLM
# backend never sees more than MAX_CONCURRENT_SUMMARIES concurrent prompts.
# Rebuilt only when the running event loop changes (tests).
_global_sem: _SummaryLimiter | None = None
_global_sem_loop: int | None = None


def _global_summary_sem() -> _SummaryLimiter:
    global _global_sem, _global_sem_loop
    loop_id = id(asyncio.get_running_loop())
    if _global_sem is None or _global_sem_loop != loop_id:
        _global_sem = _SummaryLimiter()
        _global_sem_loop = loop_id
    return _global_sem


@app.get("/healthz")
async def healthz() -> dict:
    return {
        "status": "ok",
        "llm_model": config.LLM_MODEL,
        "llm_base_url": config.LLM_BASE_URL,
        "exa_configured": bool(config.EXA_API_KEY),
        "cache": cache_mod.stats(),
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
async def post_config(req: Request, _: None = Depends(_check_admin_write)) -> JSONResponse:
    try:
        updates = await req.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
    if not isinstance(updates, dict):
        raise HTTPException(status_code=400, detail="Body must be a JSON object")
    allowed = {k: v for k, v in updates.items() if k in config.EDITABLE_FIELDS}
    errors = config.validate_updates(allowed)
    if errors:
        # all-or-nothing: nothing is applied when any field is invalid
        raise HTTPException(
            status_code=400, detail="; ".join(f"{k}: {msg}" for k, msg in errors.items())
        )
    applied = config.update_config(allowed)
    if "MAX_CONCURRENT_SUMMARIES" in applied:
        _global_summary_sem().wake()  # a raised cap admits queued jobs now
    return JSONResponse(content={
        "applied": applied,
        "config": config.public_config(),
    })


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


@app.get("/cache/stats")
async def cache_stats(_: None = Depends(_check_admin)) -> JSONResponse:
    """Admin-only: cache size, TTLs, hit/miss counters, backing file."""
    return JSONResponse(content=cache_mod.stats())


@app.post("/cache/clear")
async def cache_clear(_: None = Depends(_check_admin_write)) -> JSONResponse:
    """Admin-only: drop all cached entries (memory + disk)."""
    removed = cache_mod.clear()
    log.info("cache cleared: %d entries removed", removed)
    return JSONResponse(content={"cleared": removed, "cache": cache_mod.stats()})


@app.post("/search")
async def search(
    req: SearchRequest, authorization: str | None = Header(default=None)
) -> JSONResponse:
    _check_auth(authorization)
    if not req.query.strip():
        return JSONResponse(content=[])
    # One deadline for the whole request: search + summaries (incl. queueing).
    deadline = asyncio.get_running_loop().time() + config.REQUEST_TIMEOUT_SEC
    norm_query = cache_mod.normalize_query(req.query)
    fp = cache_mod.fingerprint() if config.CACHE_ENABLED else {}
    cache_key = cache_mod.cache_key(norm_query, req.count, fp) if config.CACHE_ENABLED else ""
    if config.CACHE_ENABLED:
        fresh = cache_mod.get_fresh(cache_key)
        if fresh is not None:
            log.info("cache fresh hit: %r count=%d", norm_query, req.count)
            return JSONResponse(content=fresh, headers={"X-Cache": "HIT"})
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

    items = _dedupe_items(items)
    sliced = items[: req.count]
    if config.CACHE_ENABLED:
        # Stale-window revalidation: same Exa results -> reuse cached summary
        # without any LLM call (still costs one Exa call to verify).
        stale = cache_mod.get_stale(cache_key)
        if stale is not None:
            fresh_canonical = cache_mod.canonical_exa(sliced)
            if cache_mod.exa_equal(stale.get("exa_items"), fresh_canonical):
                cache_mod.refresh(cache_key)
                log.info("cache revalidated hit: %r count=%d", norm_query, req.count)
                return JSONResponse(
                    content=stale.get("payload", []), headers={"X-Cache": "REVALIDATED"}
                )
            cache_mod.record_miss()
            log.info("cache stale mismatch, regenerating: %r", norm_query)

    # Stage 2 (link cache): on a query miss, reuse per-URL summaries whose
    # stored page text matches the fresh Exa text — even across different
    # queries. Only links needing (re)generation reach the LLM.
    # prefilled[i] holds (SearchResult, debug) in sliced order; pending
    # lists (idx, item, link_key, url_norm, canon_one).
    prefilled: list[tuple[SearchResult, dict] | None] = [None] * len(sliced)
    pending: list[tuple[int, dict, str, str, dict]] = []
    link_stage = config.CACHE_ENABLED and config.CACHE_LINK_ENABLED
    link_hits = 0
    if link_stage:
        for i, it in enumerate(sliced):
            url = it.get("url") or it.get("link") or ""
            url_norm = cache_mod.normalize_url(url)
            if not url_norm:
                pending.append((i, it, "", "", {}))
                continue
            lkey = cache_mod.link_key(url_norm, fp)
            canon_one = cache_mod.canonical_exa([it])[0]
            snippet, source = cache_mod.get_link_hit(lkey, canon_one)
            if snippet is not None:
                link_hits += 1
                prefilled[i] = (
                    SearchResult(link=url, title=it.get("title"), snippet=snippet),
                    {"snippet_source": source or "llm", "link_key": lkey, "cacheable": True},
                )
            else:
                pending.append((i, it, lkey, url_norm, canon_one))
    else:
        pending = [(i, it, "", "", {}) for i, it in enumerate(sliced)]
    if link_stage:
        log.info("link cache: %d/%d hits for %r", link_hits, len(sliced), norm_query)

    def _store_link(j: int, result: tuple[SearchResult, dict]) -> None:
        # Runs as each summary finishes, so finished work is cached even if
        # the request deadline later cuts off slower results.
        _, _, lkey, url_norm, canon_one = pending[j]
        res, dbg = result
        if link_stage and lkey and res.link and res.snippet and dbg.get("cacheable"):
            cache_mod.put_link(lkey, url_norm, fp, res.snippet, dbg.get("snippet_source"), canon_one)

    done = await _summarize_all(
        req.query,
        [it for _, it, _, _, _ in pending],
        deadline - asyncio.get_running_loop().time(),
        on_done=_store_link,
    )
    for (i, *_), pair in zip(pending, done):
        prefilled[i] = pair
    pairs = [p for p in prefilled if p is not None]

    # drop items with no URL; OpenWebUI tolerates fewer than `count`
    payload = [r.model_dump() for r, _ in pairs if r.link]
    if config.CACHE_ENABLED and payload:
        sources = {dbg.get("snippet_source") for _, dbg in pairs}
        if all(dbg.get("cacheable") for _, dbg in pairs):
            cache_mod.put(
                cache_key, norm_query, req.count, fp, payload,
                cache_mod.canonical_exa(sliced),
            )
            log.info("cache store: %r count=%d sources=%s", norm_query, req.count, sources)
        else:
            log.info("cache skip (LLM failure/timeout): %r sources=%s", norm_query, sources)
        headers = {"X-Cache": "MISS"}
        if link_stage:
            headers["X-Link-Cache"] = f"{link_hits}/{len(sliced)}"
        return JSONResponse(content=payload, headers=headers)
    return JSONResponse(content=payload)


@app.post("/debug/search")
async def debug_search(req: SearchRequest, _: None = Depends(_check_admin_write)) -> JSONResponse:
    """Admin-only: full pipeline trace per result.

    Returns {"mode": ..., "results": [{link, title, snippet, snippet_source,
    exa_text, exa_highlights, image_urls, comparison_summary}]} so the admin
    UI can show the raw original next to exactly what OpenWebUI receives.
    `comparison_summary` is a normal-style summary computed with the same
    rules as the Summary mode (null when already in summary/original modes
    with nothing to compare). There are no frozen prompt copies anywhere:
    every pane is produced from the live MODE rules.

    Always bypasses the result cache (live pipeline for testing).
    """
    if not req.query.strip():
        return JSONResponse(content={"results": []})
    deadline = asyncio.get_running_loop().time() + config.REQUEST_TIMEOUT_SEC
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
    items = _dedupe_items(items)
    pairs = await _summarize_all(
        req.query, items[: req.count], deadline - asyncio.get_running_loop().time()
    )
    results = [dbg for r, dbg in pairs if r.link]
    if config.MODE == "summary":
        for dbg in results:
            dbg["comparison_summary"] = None
    else:
        # One extra normal-style summary per result so the UI can compare
        # the current mode's output against the Summary mode rules.
        # Skipped when the result was already chunked: the joined parts ARE
        # the comparison, and a full-text call would defeat chunking's
        # purpose on small models. Reuses the global single-flight semaphore.
        async def _comparison(dbg: dict) -> None:
            if dbg.get("part_summaries"):
                dbg["comparison_summary"] = None
                return
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
