"""Regression tests for timeout handling, cacheability, admin hardening,
config validation and the summary limiter (all external calls mocked)."""
from __future__ import annotations

import asyncio
import base64
import importlib
import os
import time

import pytest
import respx
from fastapi.testclient import TestClient
from httpx import Response

_VARS = ("MODE", "SUMMARY_MIN_CHARS", "CHUNKED_SUMMARY", "CHUNK_MIN_CHARS", "CHUNK_TARGET_CHARS",
         "CACHE_ENABLED", "CACHE_LINK_ENABLED", "CACHE_FRESH_SEC", "CACHE_TTL_SEC",
         "REQUEST_TIMEOUT_SEC", "MAX_CONCURRENT_SUMMARIES", "EXA_API_KEY", "LLM_BASE_URL",
         "LLM_API_KEY", "EXA_BASE_URL", "ADMIN_USER", "ADMIN_PASS", "CACHE_DB_FILE", "CACHE_FILE")


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    for v in _VARS:
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setenv("CONFIG_FILE", str(tmp_path / "config.json"))
    monkeypatch.setenv("CACHE_DB_FILE", str(tmp_path / "cache.db"))
    monkeypatch.setenv("EXA_API_KEY", "k")
    monkeypatch.setenv("LLM_BASE_URL", "http://llm:8005/v1")
    monkeypatch.setenv("SUMMARY_MIN_CHARS", "0")
    yield


def _client(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import app.cache as cache_mod
    import app.config as cfg
    import app.exa as exa_mod
    import app.main as main_mod
    import app.summarizer as sum_mod

    for mod in (cfg, exa_mod, sum_mod, cache_mod, main_mod):
        importlib.reload(mod)
    cache_mod._reset_for_tests()
    return TestClient(main_mod.app), cfg, main_mod, cache_mod


def _basic():
    return {"Authorization": "Basic " + base64.b64encode(b"admin:admin").decode()}


def _exa(results):
    return respx.post("https://api.exa.ai/search").mock(
        return_value=Response(200, json={"results": results}))


def _llm_ok(content="SUMMARY"):
    return respx.post("http://llm:8005/v1/chat/completions").mock(
        return_value=Response(200, json={"choices": [{"message": {"content": content}}]}))


# --- request deadline keeps finished work ------------------------------------

def test_deadline_keeps_finished_summaries_and_link_caches_them(monkeypatch):
    client, _, main_mod, cache_mod = _client(
        monkeypatch, REQUEST_TIMEOUT_SEC="1", MAX_CONCURRENT_SUMMARIES="5", CACHE_ENABLED="true")

    async def fake(query, title, url, text, mode=None, slot_id=None, **kw):
        await asyncio.sleep(0.01 if "fast" in url else 10)
        return "S:" + url

    monkeypatch.setattr(main_mod, "summarize_one", fake)
    with respx.mock:
        _exa([{"url": f"https://e.com/fast{i}", "title": "t", "text": "x"} for i in range(2)]
             + [{"url": "https://e.com/slow", "title": "t", "text": "x", "highlights": ["HL"]}])
        r = client.post("/search", json={"query": "q", "count": 3})
    body = r.json()
    assert [b["snippet"] for b in body] == ["S:https://e.com/fast0", "S:https://e.com/fast1", "HL"]
    s = cache_mod.stats()
    assert s["link_entries"] == 2  # finished ones cached; the timed-out one is not
    assert s["query_entries"] == 0  # degraded search is never query-cached


def test_item_exception_falls_back_instead_of_emptying_search(monkeypatch):
    client, _, main_mod, _ = _client(monkeypatch)

    async def fake(query, title, url, text, mode=None, slot_id=None, **kw):
        if "bad" in url:
            raise RuntimeError("boom")
        return "S"

    monkeypatch.setattr(main_mod, "summarize_one", fake)
    with respx.mock:
        _exa([{"url": "https://e.com/ok", "title": "t", "text": "x"},
              {"url": "https://e.com/bad", "title": "t", "text": "raw text"}])
        r = client.post("/search", json={"query": "q", "count": 2})
    assert [b["snippet"] for b in r.json()] == ["S", "raw text"]


def test_debug_search_marks_timed_out_results(monkeypatch):
    client, _, main_mod, _ = _client(monkeypatch, REQUEST_TIMEOUT_SEC="1", MODE="summary")

    async def fake(*a, **kw):
        await asyncio.sleep(10)
        return "never"

    monkeypatch.setattr(main_mod, "summarize_one", fake)
    with respx.mock:
        _exa([{"url": "https://e.com/a", "title": "t", "text": "raw"}])
        r = client.post("/debug/search", json={"query": "q", "count": 1}, headers=_basic())
    res = r.json()["results"][0]
    assert res["snippet"] == "raw" and res["fallback_reason"] == "timeout"


# --- cacheability --------------------------------------------------------------

def test_chunked_summary_with_dropped_part_not_cached(monkeypatch):
    client, _, main_mod, cache_mod = _client(
        monkeypatch, CHUNKED_SUMMARY="true", CHUNK_MIN_CHARS="1000", CHUNK_TARGET_CHARS="600",
        CACHE_ENABLED="true")

    async def fake(query, title, url, text, mode=None, slot_id=None, part=None, **kw):
        return "" if part and part[0] == 2 else f"part{part}"

    monkeypatch.setattr(main_mod, "summarize_one", fake)
    text = "\n\n".join(("para %d. " % i) * 40 for i in range(6))
    with respx.mock:
        _exa([{"url": "https://e.com/a", "title": "t", "text": text}])
        client.post("/search", json={"query": "q", "count": 1})
        r2 = client.post("/search", json={"query": "q", "count": 1})
    assert r2.headers["X-Cache"] == "MISS"
    assert cache_mod.stats()["size"] == 0


def test_textless_result_does_not_block_query_cache(monkeypatch):
    client, _, _, _ = _client(monkeypatch, CACHE_ENABLED="true")
    with respx.mock:
        exa = _exa([{"url": "https://e.com/a", "title": "t", "text": "page text"},
                    {"url": "https://e.com/pdf", "title": "t", "text": "", "highlights": ["hl"]}])
        _llm_ok()
        client.post("/search", json={"query": "q", "count": 2})
        r = client.post("/search", json={"query": "q", "count": 2})
    assert r.headers["X-Cache"] == "HIT" and exa.call_count == 1
    assert r.json()[1]["snippet"] == "hl"


def test_llm_failure_still_not_cached(monkeypatch):
    client, _, _, cache_mod = _client(monkeypatch, CACHE_ENABLED="true")
    with respx.mock:
        _exa([{"url": "https://e.com/a", "title": "t", "text": "x", "highlights": ["hl"]}])
        respx.post("http://llm:8005/v1/chat/completions").mock(return_value=Response(500))
        r = client.post("/search", json={"query": "q", "count": 1})
    assert r.json()[0]["snippet"] == "hl" and cache_mod.stats()["size"] == 0


def test_revalidated_hit_not_counted_as_miss(monkeypatch):
    client, _, _, cache_mod = _client(
        monkeypatch, CACHE_ENABLED="true", CACHE_FRESH_SEC="1", CACHE_TTL_SEC="1000")
    with respx.mock:
        _exa([{"url": "https://e.com/a", "title": "t", "text": "x"}])
        _llm_ok()
        client.post("/search", json={"query": "q", "count": 1})
        for e in cache_mod._store.values():
            e["created_at"] = time.time() - 5
        r = client.post("/search", json={"query": "q", "count": 1})
    s = cache_mod.stats()
    assert r.headers["X-Cache"] == "REVALIDATED"
    assert (s["misses"], s["hits_revalidated"]) == (1, 1)


def test_dedupe_matches_link_cache_url_normalization(monkeypatch):
    client, _, _, _ = _client(monkeypatch)
    with respx.mock:
        _exa([{"url": "https://Example.com/a/", "title": "t", "text": "x"},
              {"url": "https://example.com/a#frag", "title": "t", "text": "x"}])
        llm = _llm_ok()
        r = client.post("/search", json={"query": "q", "count": 2})
    assert len(r.json()) == 1 and llm.call_count == 1


# --- admin hardening + validation ---------------------------------------------

def test_admin_posts_require_json_content_type(monkeypatch):
    client, cfg, _, _ = _client(monkeypatch)
    body = '{"EXA_BASE_URL":"http://attacker.example/","EXA_API_KEY":"x","y":"="}'
    r = client.post("/config", content=body, headers={**_basic(), "Content-Type": "text/plain"})
    assert r.status_code == 415 and cfg.EXA_BASE_URL == "https://api.exa.ai"
    assert client.post("/cache/clear", headers=_basic()).status_code == 415


def test_base_url_change_requires_key(monkeypatch):
    client, cfg, _, _ = _client(monkeypatch, EXA_API_KEY="REAL-SECRET")
    r = client.post("/config", json={"EXA_BASE_URL": "http://attacker.example"}, headers=_basic())
    assert r.status_code == 400 and "EXA_API_KEY" in r.json()["detail"]
    assert cfg.EXA_BASE_URL == "https://api.exa.ai"
    # unchanged URL (the UI resends every field) needs no key
    r = client.post("/config", json={"EXA_BASE_URL": "https://api.exa.ai/", "LLM_BASE_URL":
                                     "http://llm:8005/v1"}, headers=_basic())
    assert r.status_code == 200
    # with the key re-entered the change is allowed
    r = client.post("/config", json={"LLM_BASE_URL": "http://other:1/v1", "LLM_API_KEY": "k2"},
                    headers=_basic())
    assert r.status_code == 200 and cfg.LLM_BASE_URL == "http://other:1/v1"


def test_invalid_numbers_rejected_all_or_nothing(monkeypatch):
    client, cfg, _, _ = _client(monkeypatch)
    r = client.post("/config", json={"REQUEST_TIMEOUT_SEC": 0, "LLM_MODEL": "m2",
                                     "CACHE_MAX_ENTRIES": "lots"}, headers=_basic())
    assert r.status_code == 400
    assert "REQUEST_TIMEOUT_SEC" in r.json()["detail"] and "CACHE_MAX_ENTRIES" in r.json()["detail"]
    assert cfg.REQUEST_TIMEOUT_SEC == 1200 and cfg.LLM_MODEL != "m2"


def test_admin_save_beats_environment_after_restart(monkeypatch):
    # env seeds a value the config file does not hold yet...
    client, cfg, _, _ = _client(monkeypatch, LLM_MODEL="from-env")
    assert cfg.LLM_MODEL == "from-env"
    # ...an admin save wins, and survives a restart with the env var still set
    assert client.post("/config", json={"LLM_MODEL": "from-ui"}, headers=_basic()).status_code == 200
    _, cfg, _, _ = _client(monkeypatch)  # reload = restart
    assert os.environ["LLM_MODEL"] == "from-env" and cfg.LLM_MODEL == "from-ui"
    # keys never saved in the file still come from the environment
    assert cfg.EXA_API_KEY == "k"


def test_default_creds_reported(monkeypatch):
    client, _, _, _ = _client(monkeypatch)
    assert client.get("/config", headers=_basic()).json()["_meta"]["default_admin_creds"] is True


# --- summary limiter -----------------------------------------------------------

def _peak(main_mod, cfg, start: int, new: int, jobs: int) -> int:
    async def run():
        live = peak = 0

        async def job():
            nonlocal live, peak
            async with main_mod._global_summary_sem():
                live += 1
                peak = max(peak, live)
                await asyncio.sleep(0.05)
                live -= 1

        cfg.MAX_CONCURRENT_SUMMARIES = start
        first = asyncio.create_task(job())
        await asyncio.sleep(0.01)
        cfg.MAX_CONCURRENT_SUMMARIES = new
        main_mod._global_summary_sem().wake()
        await asyncio.gather(first, *(job() for _ in range(jobs)))
        return peak

    return asyncio.run(run())


def test_limiter_resize_never_exceeds_new_cap(monkeypatch):
    _, cfg, main_mod, _ = _client(monkeypatch)
    assert _peak(main_mod, cfg, 1, 2, 4) == 2  # used to reach 3
    assert _peak(main_mod, cfg, 2, 1, 4) == 1


def test_limiter_cancelled_waiter_releases_nothing(monkeypatch):
    _, cfg, main_mod, _ = _client(monkeypatch, MAX_CONCURRENT_SUMMARIES="1")

    async def run():
        lim = main_mod._global_summary_sem()
        async with lim:
            waiter = asyncio.create_task(lim.__aenter__())
            await asyncio.sleep(0)
            waiter.cancel()
            await asyncio.gather(waiter, return_exceptions=True)
        async with lim:  # slot must be free again, not leaked
            return lim._active

    assert asyncio.run(run()) == 1
