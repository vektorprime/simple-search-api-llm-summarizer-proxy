"""Tests for the optional exact-match (case-insensitive) result cache."""
from __future__ import annotations

import base64
import json
import time

import respx
from fastapi.testclient import TestClient
from httpx import Response


def _basic(user="admin", pw="admin"):
    tok = base64.b64encode(f"{user}:{pw}".encode()).decode()
    return {"Authorization": f"Basic {tok}"}


def _client_with_cache(tmp_path, **env_overrides):
    import os
    import importlib

    base = {
        "EXA_API_KEY": "test-exa",
        "PROXY_API_KEY": "",
        "LLM_BASE_URL": "http://llm:8005/v1",
        "LLM_MODEL": "Muse-Glimmer-30B",
        "SUMMARY_MIN_CHARS": "0",
        "ADMIN_USER": "admin",
        "ADMIN_PASS": "admin",
        "CACHE_ENABLED": "true",
        "CACHE_TTL_SEC": "172800",
        "CACHE_FRESH_SEC": "86400",
        "CACHE_MAX_ENTRIES": "200",
        "CACHE_FILE": str(tmp_path / "cache.json"),
        "CACHE_DB_FILE": str(tmp_path / "cache.db"),
        "CACHE_LINK_ENABLED": "true",
    }
    base.update(env_overrides)
    os.environ.update(base)

    import app.config as cfg
    importlib.reload(cfg)
    import app.cache as cache_mod
    importlib.reload(cache_mod)
    cache_mod._reset_for_tests()
    import app.exa as exa_mod
    import app.main as main_mod
    import app.summarizer as sum_mod
    importlib.reload(exa_mod)
    importlib.reload(sum_mod)
    importlib.reload(main_mod)
    # main re-imports cache on reload, so re-fetch the bound module
    cache_mod = main_mod.cache_mod
    cache_mod._reset_for_tests()
    return TestClient(main_mod.app), cfg, cache_mod, main_mod


def _exa_result(url="https://example.com/a", title="A", text="Full text.",
                highlights=None, image=None):
    r = {"url": url, "title": title, "text": text,
         "highlights": highlights or ["hl1"]}
    if image is not None:
        r["image"] = image
    return r


def _teardown_cache_env(*vars):
    import os
    import importlib
    for v in vars:
        os.environ.pop(v, None)
    import app.config as cfg
    import app.cache as cache_mod
    importlib.reload(cfg)
    importlib.reload(cache_mod)
    cache_mod._reset_for_tests()
    import app.exa as exa_mod
    import app.summarizer as sum_mod
    import app.main as main_mod
    importlib.reload(exa_mod)
    importlib.reload(sum_mod)
    importlib.reload(main_mod)


def test_cache_disabled_by_default():
    import os
    for v in ("CACHE_ENABLED", "CACHE_TTL_SEC", "CACHE_FRESH_SEC",
              "CACHE_MAX_ENTRIES", "CACHE_FILE", "CACHE_LINK_ENABLED", "CACHE_DB_FILE"):
        os.environ.pop(v, None)
    try:
        import importlib
        import app.config as cfg
        importlib.reload(cfg)
        assert cfg.CACHE_ENABLED is False
        assert cfg.CACHE_TTL_SEC == 172800
        assert cfg.CACHE_FRESH_SEC == 86400
        assert cfg.CACHE_MAX_ENTRIES == 200
        assert cfg.CACHE_FILE.endswith("cache.json")
        assert cfg.CACHE_DB_FILE.endswith("cache.db")
        assert cfg.CACHE_LINK_ENABLED is True
        assert cfg.update_config({"CACHE_ENABLED": "true"}) == ["CACHE_ENABLED"]
        assert cfg.CACHE_ENABLED is True
    finally:
        _teardown_cache_env("CACHE_ENABLED")


def test_fresh_hit_skips_exa_and_llm_and_is_case_insensitive(tmp_path):
    client, _, cache_mod, _ = _client_with_cache(tmp_path)
    try:
        with respx.mock:
            exa = respx.post("https://api.exa.ai/search").mock(
                return_value=Response(200, json={"results": [
                    _exa_result(text="Body about Paris.")]})
            )
            llm = respx.post("http://llm:8005/v1/chat/completions").mock(
                return_value=Response(200, json={"choices": [
                    {"message": {"content": "Summary P"}}]})
            )
            r1 = client.post("/search", json={"query": "Paris", "count": 1})
            assert r1.status_code == 200
            assert r1.json()[0]["snippet"] == "Summary P"
            assert r1.headers.get("X-Cache") == "MISS"
            assert exa.call_count == 1 and llm.call_count == 1
            # identical (different case) -> fresh hit, zero new calls
            r2 = client.post("/search", json={"query": "paris", "count": 1})
            assert r2.status_code == 200
            assert r2.json() == r1.json()
            assert r2.headers.get("X-Cache") == "HIT"
            assert exa.call_count == 1 and llm.call_count == 1
            # whitespace + case variant also hits (strip + lower)
            r3 = client.post("/search", json={"query": "  PARIS  ", "count": 1})
            assert r3.json() == r1.json()
            assert r3.headers.get("X-Cache") == "HIT"
            assert exa.call_count == 1 and llm.call_count == 1
    finally:
        _teardown_cache_env("EXA_API_KEY", "PROXY_API_KEY", "LLM_BASE_URL",
                            "LLM_MODEL", "SUMMARY_MIN_CHARS", "ADMIN_USER",
                            "ADMIN_PASS", "CACHE_ENABLED", "CACHE_TTL_SEC",
                            "CACHE_FRESH_SEC", "CACHE_MAX_ENTRIES", "CACHE_FILE", "CACHE_LINK_ENABLED", "CACHE_DB_FILE")


def test_fingerprint_is_mode_plus_images_only(tmp_path):
    client, _, cache_mod, _ = _client_with_cache(
        tmp_path, MODE="summary", RETURN_IMAGE_URLS="false")
    try:
        with respx.mock:
            exa = respx.post("https://api.exa.ai/search").mock(
                return_value=Response(200, json={"results": [
                    _exa_result(text="Body.")]})
            )
            llm = respx.post("http://llm:8005/v1/chat/completions").mock(
                return_value=Response(200, json={"choices": [
                    {"message": {"content": "S"}}]})
            )
            assert client.post("/search", json={"query": "q", "count": 1}).headers["X-Cache"] == "MISS"
            assert client.post("/search", json={"query": "q", "count": 1}).headers["X-Cache"] == "HIT"
            assert exa.call_count == 1
            # unrelated setting change (model) does NOT invalidate: still HIT
            import os
            os.environ["LLM_MODEL"] = "other-model"
            import importlib
            import app.config as cfg
            importlib.reload(cfg)
            # cache_mod fingerprint reads live config; main's fp also re-reads per request
            assert client.post("/search", json={"query": "q", "count": 1}).headers["X-Cache"] == "HIT"
            assert exa.call_count == 1  # no new Exa call
            os.environ.pop("LLM_MODEL", None)
            importlib.reload(cfg)
            # MODE change invalidates
            os.environ["MODE"] = "summary-caveman"
            importlib.reload(cfg)
            llm.mock(side_effect=[
                Response(200, json={"choices": [{"message": {"content": "CAVE"}}]}),
            ])
            r = client.post("/search", json={"query": "q", "count": 1})
            assert r.headers.get("X-Cache") == "MISS"
            assert r.json()[0]["snippet"] == "CAVE"
            os.environ.pop("MODE", None)
            importlib.reload(cfg)
    finally:
        _teardown_cache_env("EXA_API_KEY", "PROXY_API_KEY", "LLM_BASE_URL",
                            "LLM_MODEL", "SUMMARY_MIN_CHARS", "ADMIN_USER",
                            "ADMIN_PASS", "CACHE_ENABLED", "CACHE_TTL_SEC",
                            "CACHE_FRESH_SEC", "CACHE_MAX_ENTRIES", "CACHE_FILE", "CACHE_LINK_ENABLED", "CACHE_DB_FILE",
                            "MODE", "RETURN_IMAGE_URLS")


def test_stale_revalidated_hit_skips_llm_but_runs_exa(tmp_path):
    client, _, cache_mod, _ = _client_with_cache(tmp_path)
    try:
        with respx.mock:
            exa = respx.post("https://api.exa.ai/search").mock(
                return_value=Response(200, json={"results": [
                    _exa_result(text="Stable body.")]})
            )
            llm = respx.post("http://llm:8005/v1/chat/completions").mock(
                return_value=Response(200, json={"choices": [
                    {"message": {"content": "Cached summary"}}]})
            )
            r1 = client.post("/search", json={"query": "stable", "count": 1})
            assert r1.headers["X-Cache"] == "MISS"
            # age the entry into the stale window (fresh=86400, ttl=172800)
            key = cache_mod.cache_key("stable", 1, cache_mod.fingerprint())
            cache_mod._store[key]["created_at"] = time.time() - 86400 - 10
            r2 = client.post("/search", json={"query": "stable", "count": 1})
            assert r2.status_code == 200
            assert r2.json() == r1.json()
            assert r2.headers.get("X-Cache") == "REVALIDATED"
            assert exa.call_count == 2  # Exa re-ran to verify
            assert llm.call_count == 1  # LLM skipped
            # after revalidation the entry is fresh again: next hit needs no Exa
            r3 = client.post("/search", json={"query": "stable", "count": 1})
            assert r3.headers.get("X-Cache") == "HIT"
            assert exa.call_count == 2 and llm.call_count == 1
    finally:
        _teardown_cache_env("EXA_API_KEY", "PROXY_API_KEY", "LLM_BASE_URL",
                            "LLM_MODEL", "SUMMARY_MIN_CHARS", "ADMIN_USER",
                            "ADMIN_PASS", "CACHE_ENABLED", "CACHE_TTL_SEC",
                            "CACHE_FRESH_SEC", "CACHE_MAX_ENTRIES", "CACHE_FILE", "CACHE_LINK_ENABLED", "CACHE_DB_FILE")


def test_stale_mismatch_regenerates(tmp_path):
    client, _, cache_mod, _ = _client_with_cache(tmp_path)
    try:
        with respx.mock:
            exa = respx.post("https://api.exa.ai/search").mock(
                side_effect=[
                    Response(200, json={"results": [_exa_result(text="v1 body.")]}),
                    Response(200, json={"results": [_exa_result(text="v2 body CHANGED.")]}),
                ]
            )
            llm = respx.post("http://llm:8005/v1/chat/completions").mock(
                side_effect=[
                    Response(200, json={"choices": [{"message": {"content": "S1"}}]}),
                    Response(200, json={"choices": [{"message": {"content": "S2"}}]}),
                ]
            )
            r1 = client.post("/search", json={"query": "drift", "count": 1})
            assert r1.json()[0]["snippet"] == "S1"
            key = cache_mod.cache_key("drift", 1, cache_mod.fingerprint())
            cache_mod._store[key]["created_at"] = time.time() - 86400 - 10
            r2 = client.post("/search", json={"query": "drift", "count": 1})
            assert r2.headers.get("X-Cache") == "MISS"
            assert r2.json()[0]["snippet"] == "S2"
            assert exa.call_count == 2 and llm.call_count == 2
    finally:
        _teardown_cache_env("EXA_API_KEY", "PROXY_API_KEY", "LLM_BASE_URL",
                            "LLM_MODEL", "SUMMARY_MIN_CHARS", "ADMIN_USER",
                            "ADMIN_PASS", "CACHE_ENABLED", "CACHE_TTL_SEC",
                            "CACHE_FRESH_SEC", "CACHE_MAX_ENTRIES", "CACHE_FILE", "CACHE_LINK_ENABLED", "CACHE_DB_FILE")


def test_expired_entry_regenerates_without_comparison(tmp_path):
    client, _, cache_mod, _ = _client_with_cache(tmp_path)
    try:
        with respx.mock:
            exa = respx.post("https://api.exa.ai/search").mock(
                return_value=Response(200, json={"results": [
                    _exa_result(text="Same body.")]})
            )
            llm = respx.post("http://llm:8005/v1/chat/completions").mock(
                side_effect=[
                    Response(200, json={"choices": [{"message": {"content": "S1"}}]}),
                    Response(200, json={"choices": [{"message": {"content": "S2"}}]}),
                ]
            )
            assert client.post("/search", json={"query": "old", "count": 1}).json()[0]["snippet"] == "S1"
            key = cache_mod.cache_key("old", 1, cache_mod.fingerprint())
            cache_mod._store[key]["created_at"] = time.time() - 172800 - 10
            lkey = cache_mod.link_key(
                cache_mod.normalize_url("https://example.com/a"), cache_mod.fingerprint())
            cache_mod._store[lkey]["created_at"] = time.time() - 172800 - 10
            r2 = client.post("/search", json={"query": "old", "count": 1})
            assert r2.headers.get("X-Cache") == "MISS"
            assert r2.json()[0]["snippet"] == "S2"  # regenerated, not revalidated
            assert exa.call_count == 2 and llm.call_count == 2
    finally:
        _teardown_cache_env("EXA_API_KEY", "PROXY_API_KEY", "LLM_BASE_URL",
                            "LLM_MODEL", "SUMMARY_MIN_CHARS", "ADMIN_USER",
                            "ADMIN_PASS", "CACHE_ENABLED", "CACHE_TTL_SEC",
                            "CACHE_FRESH_SEC", "CACHE_MAX_ENTRIES", "CACHE_FILE", "CACHE_LINK_ENABLED", "CACHE_DB_FILE")


def test_sqlite_persistence_survives_in_memory_reset(tmp_path):
    """Rows persist in cache.db; a 'reboot' serves HIT with zero Exa/LLM calls."""
    import os
    import sqlite3
    client, _, cache_mod, _ = _client_with_cache(tmp_path)
    try:
        db_file = os.environ["CACHE_DB_FILE"]
        with respx.mock:
            respx.post("https://api.exa.ai/search").mock(
                return_value=Response(200, json={"results": [
                    _exa_result(text="Persist me.")]})
            )
            respx.post("http://llm:8005/v1/chat/completions").mock(
                return_value=Response(200, json={"choices": [
                    {"message": {"content": "P"}}]})
            )
            r1 = client.post("/search", json={"query": "persist", "count": 1})
            assert r1.headers["X-Cache"] == "MISS"
        assert os.path.exists(db_file)
        conn = sqlite3.connect(db_file)
        try:
            rows = conn.execute("SELECT key, kind, data FROM entries").fetchall()
        finally:
            conn.close()
        by_kind = {}
        for key, kind, blob in rows:
            assert key.startswith(("q:", "u:"))
            by_kind.setdefault(kind, []).append(json.loads(blob))
        assert len(by_kind["query"]) == 1 and len(by_kind["link"]) == 1
        assert by_kind["query"][0]["payload"][0]["snippet"] == "P"
        assert by_kind["query"][0]["exa_items"][0]["text"] == "Persist me."
        assert by_kind["link"][0]["snippet"] == "P"
        # simulate reboot: drop memory, reload from db
        cache_mod._reset_for_tests()
        with respx.mock:
            exa2 = respx.post("https://api.exa.ai/search").mock(
                return_value=Response(200, json={"results": []})
            )
            llm2 = respx.post("http://llm:8005/v1/chat/completions").mock(
                return_value=Response(200, json={"choices": [
                    {"message": {"content": "MUST NOT BE CALLED"}}]})
            )
            r2 = client.post("/search", json={"query": "persist", "count": 1})
            assert r2.json() == r1.json()
            assert r2.headers.get("X-Cache") == "HIT"
            assert exa2.call_count == 0 and llm2.call_count == 0
    finally:
        _teardown_cache_env("EXA_API_KEY", "PROXY_API_KEY", "LLM_BASE_URL",
                            "LLM_MODEL", "SUMMARY_MIN_CHARS", "ADMIN_USER",
                            "ADMIN_PASS", "CACHE_ENABLED", "CACHE_TTL_SEC",
                            "CACHE_FRESH_SEC", "CACHE_MAX_ENTRIES", "CACHE_FILE", "CACHE_LINK_ENABLED", "CACHE_DB_FILE")


def test_migration_from_legacy_json(tmp_path):
    """A legacy cache.json is imported once (keys recomputed), then served."""
    import os
    client, _, cache_mod, _ = _client_with_cache(tmp_path)
    try:
        fp = cache_mod.fingerprint()  # live config (repo config.json may leak in)
        payload = [{"link": "https://example.com/m", "title": "M", "snippet": "Old summary"}]
        exa_items = [{"url": "https://example.com/m", "title": "M",
                      "text": "Old body.", "highlights": ["hl"], "image": None}]
        legacy = {"version": 2, "entries": {
            "bare-old-key": {
                "query_norm": "migrated query", "count": 1, "fingerprint": fp,
                "created_at": time.time(),
                "payload": payload, "exa_items": exa_items,
            },
            "bare-old-link": {
                "url_norm": "https://example.com/m", "fingerprint": fp,
                "created_at": time.time(),
                "snippet": "Old summary", "source": "llm", "exa": exa_items[0],
            },
        }}
        with open(os.environ["CACHE_FILE"], "w") as f:
            json.dump(legacy, f)
        with respx.mock:
            exa = respx.post("https://api.exa.ai/search").mock(
                return_value=Response(200, json={"results": []})
            )
            llm = respx.post("http://llm:8005/v1/chat/completions").mock(
                return_value=Response(200, json={"choices": [
                    {"message": {"content": "MUST NOT BE CALLED"}}]})
            )
            r = client.post("/search", json={"query": "migrated query", "count": 1})
            assert r.headers.get("X-Cache") == "HIT"  # served from migrated row
            assert r.json() == payload
            assert exa.call_count == 0 and llm.call_count == 0
            st = cache_mod.stats()
            assert st["query_entries"] == 1 and st["link_entries"] == 1
    finally:
        _teardown_cache_env("EXA_API_KEY", "PROXY_API_KEY", "LLM_BASE_URL",
                            "LLM_MODEL", "SUMMARY_MIN_CHARS", "ADMIN_USER",
                            "ADMIN_PASS", "CACHE_ENABLED", "CACHE_TTL_SEC",
                            "CACHE_FRESH_SEC", "CACHE_MAX_ENTRIES", "CACHE_FILE", "CACHE_LINK_ENABLED", "CACHE_DB_FILE")


def test_fallback_results_are_not_cached(tmp_path):
    client, _, cache_mod, _ = _client_with_cache(tmp_path)
    try:
        with respx.mock:
            exa = respx.post("https://api.exa.ai/search").mock(
                return_value=Response(200, json={"results": [
                    _exa_result(text="Body.", highlights=["hl-fallback"])]})
            )
            llm = respx.post("http://llm:8005/v1/chat/completions").mock(
                return_value=Response(500, json={"error": "boom"})
            )
            r1 = client.post("/search", json={"query": "flaky", "count": 1})
            assert "hl-fallback" in r1.json()[0]["snippet"]
            assert r1.headers.get("X-Cache") == "MISS"
            assert cache_mod.stats()["size"] == 0  # nothing stored
            r2 = client.post("/search", json={"query": "flaky", "count": 1})
            assert r2.headers.get("X-Cache") == "MISS"
            assert exa.call_count == 2 and llm.call_count == 2
    finally:
        _teardown_cache_env("EXA_API_KEY", "PROXY_API_KEY", "LLM_BASE_URL",
                            "LLM_MODEL", "SUMMARY_MIN_CHARS", "ADMIN_USER",
                            "ADMIN_PASS", "CACHE_ENABLED", "CACHE_TTL_SEC",
                            "CACHE_FRESH_SEC", "CACHE_MAX_ENTRIES", "CACHE_FILE", "CACHE_LINK_ENABLED", "CACHE_DB_FILE")


def test_original_mode_is_cached(tmp_path):
    client, _, cache_mod, _ = _client_with_cache(
        tmp_path, MODE="original", SUMMARY_MIN_CHARS="0")
    try:
        with respx.mock:
            exa = respx.post("https://api.exa.ai/search").mock(
                return_value=Response(200, json={"results": [
                    _exa_result(text="Raw verbatim.")]})
            )
            llm = respx.post("http://llm:8005/v1/chat/completions").mock(
                return_value=Response(200, json={"choices": [
                    {"message": {"content": "MUST NOT BE CALLED"}}]})
            )
            r1 = client.post("/search", json={"query": "raw", "count": 1})
            assert r1.json()[0]["snippet"] == "Raw verbatim."
            assert llm.call_count == 0
            r2 = client.post("/search", json={"query": "RAW", "count": 1})
            assert r2.json() == r1.json()
            assert r2.headers.get("X-Cache") == "HIT"
            assert exa.call_count == 1
    finally:
        _teardown_cache_env("EXA_API_KEY", "PROXY_API_KEY", "LLM_BASE_URL",
                            "LLM_MODEL", "SUMMARY_MIN_CHARS", "ADMIN_USER",
                            "ADMIN_PASS", "CACHE_ENABLED", "CACHE_TTL_SEC",
                            "CACHE_FRESH_SEC", "CACHE_MAX_ENTRIES", "CACHE_FILE", "CACHE_LINK_ENABLED", "CACHE_DB_FILE",
                            "MODE")


def test_debug_search_bypasses_cache_and_stats_clear_require_auth(tmp_path):
    client, _, cache_mod, _ = _client_with_cache(tmp_path)
    try:
        assert client.get("/cache/stats").status_code == 401
        assert client.post("/cache/clear").status_code == 401
        with respx.mock:
            respx.post("https://api.exa.ai/search").mock(
                return_value=Response(200, json={"results": [
                    _exa_result(text="Body.")]})
            )
            respx.post("http://llm:8005/v1/chat/completions").mock(
                return_value=Response(200, json={"choices": [
                    {"message": {"content": "S"}}]})
            )
            assert client.post("/search", json={"query": "q", "count": 1}).headers["X-Cache"] == "MISS"
            # debug runs the live pipeline (no X-Cache header, no hit counting)
            r = client.post("/debug/search", json={"query": "q", "count": 1}, headers=_basic())
            assert r.status_code == 200
            assert "X-Cache" not in r.headers
            s = client.get("/cache/stats", headers=_basic())
            assert s.status_code == 200
            assert s.json()["size"] == 2  # 1 query + 1 link entry
            assert s.json()["query_entries"] == 1
            assert s.json()["link_entries"] == 1
            assert s.json()["enabled"] is True
            c = client.post("/cache/clear", headers=_basic())
            assert c.status_code == 200
            assert c.json()["cleared"] == 2
            assert client.get("/cache/stats", headers=_basic()).json()["size"] == 0
    finally:
        _teardown_cache_env("EXA_API_KEY", "PROXY_API_KEY", "LLM_BASE_URL",
                            "LLM_MODEL", "SUMMARY_MIN_CHARS", "ADMIN_USER",
                            "ADMIN_PASS", "CACHE_ENABLED", "CACHE_TTL_SEC",
                            "CACHE_FRESH_SEC", "CACHE_MAX_ENTRIES", "CACHE_FILE", "CACHE_LINK_ENABLED", "CACHE_DB_FILE")


def test_normalize_strips_quotes_and_separator_punctuation():
    from app.cache import normalize_query
    assert normalize_query("Paris") == "paris"
    assert normalize_query("  PARIS  ") == "paris"
    assert normalize_query('"Paris"') == "paris"
    assert normalize_query("“Paris”") == "paris"
    assert normalize_query("  'Paris!'  ") == "paris"
    assert normalize_query("study: med") == normalize_query("study med")
    assert normalize_query("study, med;") == normalize_query("study med")
    assert normalize_query("What is X?") != normalize_query("What is X")  # ? not in set
    assert normalize_query("a  b\tc") == "a b c"
    assert normalize_query("") == ""


def test_quoted_and_punctuated_repeat_hits_cache(tmp_path):
    client, _, _, _ = _client_with_cache(tmp_path)
    try:
        with respx.mock:
            exa = respx.post("https://api.exa.ai/search").mock(
                return_value=Response(200, json={"results": [
                    _exa_result(text="Body about Paris.")]})
            )
            llm = respx.post("http://llm:8005/v1/chat/completions").mock(
                return_value=Response(200, json={"choices": [
                    {"message": {"content": "Summary P"}}]})
            )
            r1 = client.post("/search", json={"query": "Efficacy study: med", "count": 1})
            assert r1.headers["X-Cache"] == "MISS"
            # curly quotes + case + extra spaces + trailing ! all normalize away
            r2 = client.post("/search", json={"query": "“efficacy  study med!”", "count": 1})
            assert r2.json() == r1.json()
            assert r2.headers.get("X-Cache") == "HIT"
            assert exa.call_count == 1 and llm.call_count == 1
    finally:
        _teardown_cache_env("EXA_API_KEY", "PROXY_API_KEY", "LLM_BASE_URL",
                            "LLM_MODEL", "SUMMARY_MIN_CHARS", "ADMIN_USER",
                            "ADMIN_PASS", "CACHE_ENABLED", "CACHE_TTL_SEC",
                            "CACHE_FRESH_SEC", "CACHE_MAX_ENTRIES", "CACHE_FILE", "CACHE_LINK_ENABLED", "CACHE_DB_FILE")


def test_count_is_part_of_key(tmp_path):
    client, _, _, _ = _client_with_cache(tmp_path)
    try:
        with respx.mock:
            exa = respx.post("https://api.exa.ai/search").mock(
                return_value=Response(200, json={"results": [
                    _exa_result(url="https://example.com/a", text="A."),
                    _exa_result(url="https://example.com/b", text="B."),
                ]})
            )
            llm = respx.post("http://llm:8005/v1/chat/completions").mock(
                return_value=Response(200, json={"choices": [
                    {"message": {"content": "S"}}]})
            )
            assert client.post("/search", json={"query": "q", "count": 1}).headers["X-Cache"] == "MISS"
            # same query, different count -> MISS (not a slice hit)
            assert client.post("/search", json={"query": "q", "count": 2}).headers["X-Cache"] == "MISS"
            assert exa.call_count == 2
    finally:
        _teardown_cache_env("EXA_API_KEY", "PROXY_API_KEY", "LLM_BASE_URL",
                            "LLM_MODEL", "SUMMARY_MIN_CHARS", "ADMIN_USER",
                            "ADMIN_PASS", "CACHE_ENABLED", "CACHE_TTL_SEC",
                            "CACHE_FRESH_SEC", "CACHE_MAX_ENTRIES", "CACHE_FILE", "CACHE_LINK_ENABLED", "CACHE_DB_FILE")


def _link_teardown():
    _teardown_cache_env("EXA_API_KEY", "PROXY_API_KEY", "LLM_BASE_URL",
                        "LLM_MODEL", "SUMMARY_MIN_CHARS", "ADMIN_USER",
                        "ADMIN_PASS", "CACHE_ENABLED", "CACHE_TTL_SEC",
                        "CACHE_FRESH_SEC", "CACHE_MAX_ENTRIES", "CACHE_FILE", "CACHE_LINK_ENABLED", "CACHE_DB_FILE",
                        "MODE")


def test_normalize_url():
    from app.cache import normalize_url
    assert normalize_url("https://example.com/a/") == "https://example.com/a"
    assert normalize_url("https://Example.com/a") == "https://example.com/a"
    assert normalize_url("HTTPS://EXAMPLE.COM/a") == "https://example.com/a"
    assert normalize_url("https://example.com/a#section") == "https://example.com/a"
    assert normalize_url("https://example.com/a?x=1") == "https://example.com/a?x=1"
    assert normalize_url("https://example.com/A") != "https://example.com/a"  # path stays case-sensitive
    assert normalize_url("") == ""
    assert normalize_url("  https://example.com/b/  ") == "https://example.com/b"


def test_link_cache_reuses_across_different_queries(tmp_path):
    """Stage 2: a different query returning the same URLs skips the LLM."""
    client, _, cache_mod, _ = _client_with_cache(tmp_path)
    try:
        shared = [_exa_result(url="https://example.com/a", title="A", text="Text A."),
                  _exa_result(url="https://example.com/b", title="B", text="Text B.")]
        with respx.mock:
            exa = respx.post("https://api.exa.ai/search").mock(
                return_value=Response(200, json={"results": shared})
            )
            llm = respx.post("http://llm:8005/v1/chat/completions").mock(
                side_effect=[
                    Response(200, json={"choices": [{"message": {"content": "Sum A"}}]}),
                    Response(200, json={"choices": [{"message": {"content": "Sum B"}}]}),
                ]
            )
            r1 = client.post("/search", json={"query": "tour guide for paris", "count": 2})
            assert r1.headers["X-Cache"] == "MISS"
            assert r1.headers.get("X-Link-Cache") == "0/2"
            assert [d["snippet"] for d in r1.json()] == ["Sum A", "Sum B"]
            assert exa.call_count == 1 and llm.call_count == 2
            # completely different query, same URLs+texts -> query MISS, links 2/2
            r2 = client.post("/search", json={"query": "paris sightseeing tours", "count": 2})
            assert r2.headers["X-Cache"] == "MISS"
            assert r2.headers.get("X-Link-Cache") == "2/2"
            assert r2.json() == r1.json()
            assert exa.call_count == 2  # Exa still ran (stage 1 missed)
            assert llm.call_count == 2  # LLM skipped for both links
            st = cache_mod.stats()
            assert st["link_entries"] == 2
            assert st["link_hits"] == 2
    finally:
        _link_teardown()


def test_link_cache_partial_regeneration(tmp_path):
    """Only links with changed page text regenerate; the rest hit."""
    client, _, _, _ = _client_with_cache(tmp_path)
    try:
        with respx.mock:
            exa = respx.post("https://api.exa.ai/search").mock(
                side_effect=[
                    Response(200, json={"results": [
                        _exa_result(url="https://example.com/a", text="Text A."),
                        _exa_result(url="https://example.com/b", text="Text B v1."),
                    ]}),
                    Response(200, json={"results": [
                        _exa_result(url="https://example.com/a", text="Text A."),
                        _exa_result(url="https://example.com/b", text="Text B v2 CHANGED."),
                    ]}),
                ]
            )
            llm = respx.post("http://llm:8005/v1/chat/completions").mock(
                side_effect=[
                    Response(200, json={"choices": [{"message": {"content": "Sum A"}}]}),
                    Response(200, json={"choices": [{"message": {"content": "Sum B1"}}]}),
                    Response(200, json={"choices": [{"message": {"content": "Sum B2"}}]}),
                ]
            )
            r1 = client.post("/search", json={"query": "first query", "count": 2})
            assert [d["snippet"] for d in r1.json()] == ["Sum A", "Sum B1"]
            r2 = client.post("/search", json={"query": "second query", "count": 2})
            assert r2.headers.get("X-Link-Cache") == "1/2"
            assert [d["snippet"] for d in r2.json()] == ["Sum A", "Sum B2"]
            assert exa.call_count == 2 and llm.call_count == 3
    finally:
        _link_teardown()


def test_link_cache_trailing_slash_variant_hits(tmp_path):
    client, _, _, _ = _client_with_cache(tmp_path)
    try:
        with respx.mock:
            exa = respx.post("https://api.exa.ai/search").mock(
                side_effect=[
                    Response(200, json={"results": [
                        _exa_result(url="https://example.com/a", text="Text.")]}),
                    Response(200, json={"results": [
                        _exa_result(url="https://example.com/a/", text="Text.")]}),
                ]
            )
            llm = respx.post("http://llm:8005/v1/chat/completions").mock(
                return_value=Response(200, json={"choices": [{"message": {"content": "S"}}]})
            )
            assert client.post("/search", json={"query": "q one", "count": 1}).headers["X-Link-Cache"] == "0/1"
            r2 = client.post("/search", json={"query": "q two", "count": 1})
            assert r2.headers.get("X-Link-Cache") == "1/1"
            assert r2.json()[0]["snippet"] == "S"
            assert llm.call_count == 1
    finally:
        _link_teardown()


def test_link_cache_mode_change_misses(tmp_path):
    client, _, _, _ = _client_with_cache(tmp_path, MODE="summary")
    try:
        import os
        import importlib
        with respx.mock:
            exa = respx.post("https://api.exa.ai/search").mock(
                return_value=Response(200, json={"results": [
                    _exa_result(url="https://example.com/a", text="Text.")]})
            )
            llm = respx.post("http://llm:8005/v1/chat/completions").mock(
                side_effect=[
                    Response(200, json={"choices": [{"message": {"content": "S1"}}]}),
                    Response(200, json={"choices": [{"message": {"content": "S2"}}]}),
                ]
            )
            assert client.post("/search", json={"query": "q one", "count": 1}).json()[0]["snippet"] == "S1"
            os.environ["MODE"] = "summary-caveman"
            import app.config as cfg
            importlib.reload(cfg)
            r2 = client.post("/search", json={"query": "q two", "count": 1})
            assert r2.headers.get("X-Link-Cache") == "0/1"  # fingerprint changed
            assert r2.json()[0]["snippet"] == "S2"
            assert llm.call_count == 2
    finally:
        _link_teardown()


def test_link_cache_expiry_regenerates(tmp_path):
    client, _, cache_mod, _ = _client_with_cache(tmp_path)
    try:
        with respx.mock:
            exa = respx.post("https://api.exa.ai/search").mock(
                return_value=Response(200, json={"results": [
                    _exa_result(url="https://example.com/a", text="Text.")]})
            )
            llm = respx.post("http://llm:8005/v1/chat/completions").mock(
                side_effect=[
                    Response(200, json={"choices": [{"message": {"content": "S1"}}]}),
                    Response(200, json={"choices": [{"message": {"content": "S2"}}]}),
                ]
            )
            assert client.post("/search", json={"query": "q one", "count": 1}).json()[0]["snippet"] == "S1"
            lkey = cache_mod.link_key(cache_mod.normalize_url("https://example.com/a"),
                                      cache_mod.fingerprint())
            cache_mod._store[lkey]["created_at"] = time.time() - 172800 - 10
            r2 = client.post("/search", json={"query": "q two", "count": 1})
            assert r2.headers.get("X-Link-Cache") == "0/1"
            assert r2.json()[0]["snippet"] == "S2"
            assert llm.call_count == 2
    finally:
        _link_teardown()


def test_link_cache_disabled_independently(tmp_path):
    client, _, cache_mod, _ = _client_with_cache(tmp_path, CACHE_LINK_ENABLED="false")
    try:
        with respx.mock:
            exa = respx.post("https://api.exa.ai/search").mock(
                return_value=Response(200, json={"results": [
                    _exa_result(url="https://example.com/a", text="Text.")]})
            )
            llm = respx.post("http://llm:8005/v1/chat/completions").mock(
                return_value=Response(200, json={"choices": [{"message": {"content": "S"}}]})
            )
            client.post("/search", json={"query": "q one", "count": 1})
            r2 = client.post("/search", json={"query": "q two", "count": 1})
            assert "X-Link-Cache" not in r2.headers  # stage 2 never ran
            assert llm.call_count == 2
            assert cache_mod.stats()["link_entries"] == 0
    finally:
        _link_teardown()
