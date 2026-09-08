"""Tests for the Exa-summary search proxy (all external calls mocked)."""
from __future__ import annotations

import base64

import pytest
import respx
from fastapi.testclient import TestClient
from httpx import Response


def _client(**env_overrides):
    import os

    os.environ.update(env_overrides)
    # fresh config import per override set
    import importlib

    import app.config as cfg

    importlib.reload(cfg)
    import app.exa as exa_mod
    import app.main as main_mod
    import app.summarizer as sum_mod

    importlib.reload(exa_mod)
    importlib.reload(sum_mod)
    importlib.reload(main_mod)
    return TestClient(main_mod.app), cfg


def _basic(user="admin", pw="admin"):
    tok = base64.b64encode(f"{user}:{pw}".encode()).decode()
    return {"Authorization": f"Basic {tok}"}


@pytest.fixture(autouse=True)
def _isolate_config_file(tmp_path, monkeypatch):
    """Keep tests independent of ./config.json / ./.env in the working directory."""
    import os as _os

    if "CONFIG_FILE" not in _os.environ:
        monkeypatch.setenv("CONFIG_FILE", str(tmp_path / "nonexistent.json"))


def test_search_happy_path_summarized():
    client, _ = _client(
        EXA_API_KEY="test-exa",
        PROXY_API_KEY="",
        LLM_BASE_URL="http://llm:8005/v1",
        LLM_MODEL="Muse-Glimmer-30B",
        LLM_MAX_TOKENS="2000",
        ADMIN_USER="admin",
        ADMIN_PASS="admin",
    )
    with respx.mock:
        respx.post("https://api.exa.ai/search").mock(
            return_value=Response(
                200,
                json={
                    "results": [
                        {
                            "url": "https://example.com/a",
                            "title": "A",
                            "text": "Full text about Paris.",
                            "highlights": ["highlight A"],
                        },
                        {
                            "url": "https://example.com/b",
                            "title": "B",
                            "text": "Full text about Rome.",
                            "highlights": ["highlight B"],
                        },
                    ]
                },
            )
        )
        llm = respx.post("http://llm:8005/v1/chat/completions").mock(
            side_effect=[
                Response(200, json={"choices": [{"message": {"content": "Summary A"}}]}),
                Response(200, json={"choices": [{"message": {"content": "Summary B"}}]}),
            ]
        )
        r = client.post("/search", json={"query": "Paris", "count": 2})
    assert r.status_code == 200, r.text
    data = r.json()
    assert [d["link"] for d in data] == ["https://example.com/a", "https://example.com/b"]
    assert data[0]["snippet"] == "Summary A"
    assert data[1]["snippet"] == "Summary B"
    assert llm.call_count == 2
    # Reasoning backends need big max_tokens; ensure we request enough
    assert llm.calls[0].request.content.count(b"max_tokens") == 1
    import json as _json

    body = _json.loads(llm.calls[0].request.content)
    assert body["max_tokens"] >= 1500


def test_search_llm_failure_falls_back_to_highlights():
    client, _ = _client(EXA_API_KEY="k", PROXY_API_KEY="")
    with respx.mock:
        respx.post("https://api.exa.ai/search").mock(
            return_value=Response(
                200,
                json={
                    "results": [
                        {
                            "url": "https://example.com/a",
                            "title": "A",
                            "text": "Some text",
                            "highlights": ["hl1", "hl2"],
                        }
                    ]
                },
            )
        )
        respx.post("http://10.0.0.187:8005/v1/chat/completions").mock(
            return_value=Response(500, json={"error": "boom"})
        )
        # also cover default LLM URL host variant used in tests
        respx.post("http://llm:8005/v1/chat/completions").mock(
            return_value=Response(500, json={"error": "boom"})
        )
        r = client.post("/search", json={"query": "q", "count": 1})
    assert r.status_code == 200
    assert "hl1" in r.json()[0]["snippet"]


def test_search_exa_failure_returns_empty_list():
    client, _ = _client(EXA_API_KEY="k", PROXY_API_KEY="")
    with respx.mock:
        respx.post("https://api.exa.ai/search").mock(
            return_value=Response(500, json={"error": "exa down"})
        )
        r = client.post("/search", json={"query": "q", "count": 3})
    assert r.status_code == 200
    assert r.json() == []


def test_search_accepts_any_bearer():
    client, _ = _client(EXA_API_KEY="k", PROXY_API_KEY="secret123")
    with respx.mock:
        respx.post("https://api.exa.ai/search").mock(
            return_value=Response(200, json={"results": []})
        )
        # no key, wrong key, any key -> all accepted (200, [] from empty results)
        assert client.post("/search", json={"query": "q", "count": 1}).status_code == 200
        r2 = client.post(
            "/search", json={"query": "q", "count": 1}, headers={"Authorization": "Bearer wrong"}
        )
        assert r2.status_code == 200
        r3 = client.post(
            "/search",
            json={"query": "q", "count": 1},
            headers={"Authorization": "Bearer secret123"},
        )
        assert r3.status_code == 200


def test_admin_requires_login_and_updates_config():
    client, cfg = _client(ADMIN_USER="admin", ADMIN_PASS="admin", LLM_MODEL="Muse-Glimmer-30B")
    r = client.get("/config")
    assert r.status_code == 401
    r = client.get("/config", headers=_basic("admin", "wrong"))
    assert r.status_code == 401
    r = client.get("/config", headers=_basic())
    assert r.status_code == 200
    assert "LLM_MODEL" in r.json()

    r = client.post(
        "/config", json={"LLM_MODEL": "other-model", "LLM_MAX_TOKENS": 2500}, headers=_basic()
    )
    assert r.status_code == 200
    assert "LLM_MODEL" in r.json()["applied"]
    assert cfg.LLM_MODEL == "other-model"

    r = client.get("/admin", headers=_basic())
    assert r.status_code == 200
    assert "Connect to OpenWebUI" in r.text
    assert "Timeouts" in r.text  # grouped settings render


def test_think_tags_stripped():
    from app.summarizer import clean_summary

    assert clean_summary("<think>hidden</think>Visible") == "Visible"
    assert clean_summary(None) == ""


def test_single_flight_defaults():
    import os

    for var in ("MAX_CONCURRENT_SUMMARIES", "LLM_TIMEOUT_SEC", "REQUEST_TIMEOUT_SEC",
                 "LLM_MAX_TOKENS"):
        os.environ.pop(var, None)
    import importlib

    import app.config as cfg

    importlib.reload(cfg)
    assert cfg.MAX_CONCURRENT_SUMMARIES == 1
    assert cfg.LLM_TIMEOUT_SEC == 300
    assert cfg.REQUEST_TIMEOUT_SEC == 1200
    assert cfg.LLM_MAX_TOKENS == 0  # unlimited by default
    assert cfg.EXA_TEXT_MAX_CHARS == 0  # unlimited by default
    assert cfg.SUMMARY_INPUT_MAX_CHARS == 0  # unlimited by default


@pytest.mark.asyncio
async def test_max_tokens_zero_omits_limit():
    import json as _json
    import os

    os.environ["LLM_BASE_URL"] = "http://llm:8005/v1"
    os.environ["LLM_MAX_TOKENS"] = "0"
    import importlib

    import app.config as cfg
    import app.summarizer as sum_mod

    importlib.reload(cfg)
    importlib.reload(sum_mod)
    with respx.mock:
        route = respx.post("http://llm:8005/v1/chat/completions").mock(
            return_value=Response(200, json={"choices": [{"message": {"content": "s"}}]})
        )
        out = await sum_mod.summarize_one("q", "t", "http://x", "some page text")
    assert out == "s"
    body = _json.loads(route.calls[0].request.content)
    assert "max_tokens" not in body
    assert "temperature" not in body


@pytest.mark.asyncio
async def test_max_tokens_positive_is_sent():
    import json as _json
    import os

    os.environ["LLM_BASE_URL"] = "http://llm:8005/v1"
    os.environ["LLM_MAX_TOKENS"] = "2000"
    import importlib

    import app.config as cfg
    import app.summarizer as sum_mod

    importlib.reload(cfg)
    importlib.reload(sum_mod)
    with respx.mock:
        route = respx.post("http://llm:8005/v1/chat/completions").mock(
            return_value=Response(200, json={"choices": [{"message": {"content": "s"}}]})
        )
        await sum_mod.summarize_one("q", "t", "http://x", "some page text")
    body = _json.loads(route.calls[0].request.content)
    assert body["max_tokens"] == 2000


async def _summarize_body(env, **kwargs):
    import json as _json
    import os

    os.environ.update(env)
    for var in ("LLM_PROVIDER", "LLAMACPP_USE_SLOTS", "LLAMACPP_SLOT_COUNT",
                "LLAMACPP_SLOT_ID", "MODE"):
        if var not in env:
            os.environ.pop(var, None)
    import importlib

    import app.config as cfg
    import app.summarizer as sum_mod

    importlib.reload(cfg)
    importlib.reload(sum_mod)
    with respx.mock:
        route = respx.post("http://llm:8005/v1/chat/completions").mock(
            return_value=Response(200, json={"choices": [{"message": {"content": "s"}}]})
        )
        await sum_mod.summarize_one("q", "t", "http://x", "some page text", **kwargs)
    return _json.loads(route.calls[0].request.content), cfg


@pytest.mark.asyncio
async def test_slot_id_sent_when_provided_omitted_otherwise():
    body, _ = await _summarize_body(
        {"LLM_BASE_URL": "http://llm:8005/v1"}, slot_id=2
    )
    assert body["id_slot"] == 2
    body, _ = await _summarize_body({"LLM_BASE_URL": "http://llm:8005/v1"})
    assert "id_slot" not in body


def _reload_main():
    import importlib

    import app.main as main_mod

    importlib.reload(main_mod)
    return main_mod


def test_slot_rotation_cycles_pool_in_order():
    import os

    os.environ.update(
        {"LLM_PROVIDER": "llamacpp", "LLAMACPP_USE_SLOTS": "true",
         "LLAMACPP_SLOT_COUNT": "2"}
    )
    os.environ.pop("LLAMACPP_SLOT_ID", None)
    try:
        import app.config as cfg
        import importlib

        importlib.reload(cfg)
        main_mod = _reload_main()
        main_mod._slot_counter = 0
        assert [main_mod._rotated_slot_id() for _ in range(5)] == [0, 1, 0, 1, 0]
    finally:
        for var in ("LLM_PROVIDER", "LLAMACPP_USE_SLOTS", "LLAMACPP_SLOT_COUNT"):
            os.environ.pop(var, None)


def test_slot_rotation_off_unless_llamacpp_and_managed():
    import os

    try:
        import app.config as cfg
        import importlib

        os.environ.update(
            {"LLM_PROVIDER": "llamacpp", "LLAMACPP_USE_SLOTS": "false",
             "LLAMACPP_SLOT_COUNT": "4"}
        )
        os.environ.pop("LLAMACPP_SLOT_ID", None)
        importlib.reload(cfg)
        main_mod = _reload_main()
        assert main_mod._rotated_slot_id() is None
        os.environ.update({"LLM_PROVIDER": "vllm", "LLAMACPP_USE_SLOTS": "true"})
        importlib.reload(cfg)
        main_mod = _reload_main()
        assert main_mod._rotated_slot_id() is None
    finally:
        for var in ("LLM_PROVIDER", "LLAMACPP_USE_SLOTS", "LLAMACPP_SLOT_COUNT",
                    "LLAMACPP_SLOT_ID"):
            os.environ.pop(var, None)


def test_llamacpp_slot_defaults_bool_parsing_and_legacy_id():
    import os

    for var in ("LLM_PROVIDER", "LLAMACPP_USE_SLOTS", "LLAMACPP_SLOT_COUNT",
                "LLAMACPP_SLOT_ID"):
        os.environ.pop(var, None)
    import importlib

    import app.config as cfg

    importlib.reload(cfg)
    assert cfg.LLM_PROVIDER == "llamacpp"
    assert cfg.LLAMACPP_USE_SLOTS is False
    assert cfg.LLAMACPP_SLOT_COUNT == 1
    assert cfg.update_config({"LLAMACPP_USE_SLOTS": "true"}) == ["LLAMACPP_USE_SLOTS"]
    assert cfg.LLAMACPP_USE_SLOTS is True
    cfg.update_config({"LLAMACPP_USE_SLOTS": "false"})
    assert cfg.LLAMACPP_USE_SLOTS is False
    cfg.update_config({"LLM_PROVIDER": "VLLM"})
    assert cfg.LLM_PROVIDER == "vllm"
    # legacy single slot id migrates to a pool covering 0..id
    os.environ["LLAMACPP_SLOT_ID"] = "2"
    try:
        importlib.reload(cfg)
        assert cfg.LLAMACPP_SLOT_COUNT == 3
    finally:
        os.environ.pop("LLAMACPP_SLOT_ID", None)


@pytest.mark.asyncio
async def test_mode_switches_prompt():
    import json as _json
    import os

    async def body_with(mode):
        os.environ["LLM_BASE_URL"] = "http://llm:8005/v1"
        os.environ["MODE"] = mode
        import importlib

        import app.config as cfg
        import app.summarizer as sum_mod

        importlib.reload(cfg)
        importlib.reload(sum_mod)
        with respx.mock:
            route = respx.post("http://llm:8005/v1/chat/completions").mock(
                return_value=Response(200, json={"choices": [{"message": {"content": "s"}}]})
            )
            await sum_mod.summarize_one("q", "t", "http://x", "some page text")
        return _json.loads(route.calls[0].request.content)

    try:
        plain = await body_with("summary")
        assert "Do not summarize or omit facts" not in plain["messages"][0]["content"]
        assert "Search query:" in plain["messages"][1]["content"]  # grounding header kept
        cave = await body_with("original-caveman")
        assert "Do not summarize or omit facts" in cave["messages"][0]["content"]
        assert "caveman" in cave["messages"][1]["content"].lower()
        assert "Search query:" not in cave["messages"][1]["content"]  # no echo surface
        assert "Page URL:" not in cave["messages"][1]["content"]
        short = await body_with("summary-caveman")
        # same task as plain summary — only the system prompt (style) differs
        assert short["messages"][1] == plain["messages"][1]
        assert short["messages"][0] != plain["messages"][0]
        assert "Do not summarize or omit facts" in short["messages"][0]["content"]
    finally:
        import os as _os

        _os.environ.pop("MODE", None)  # don't leak into other tests


@pytest.mark.asyncio
async def test_explicit_mode_overrides_config():
    """summarize_one(mode=...) beats the configured MODE — lets the test UI
    derive every pane from explicit modes instead of frozen prompt copies."""
    import json as _json
    import os

    os.environ["LLM_BASE_URL"] = "http://llm:8005/v1"
    os.environ["MODE"] = "summary"
    try:
        import importlib

        import app.config as cfg
        import app.summarizer as sum_mod

        importlib.reload(cfg)
        importlib.reload(sum_mod)
        with respx.mock:
            route = respx.post("http://llm:8005/v1/chat/completions").mock(
                return_value=Response(200, json={"choices": [{"message": {"content": "s"}}]})
            )
            await sum_mod.summarize_one("q", "t", "http://x", "text", mode="original-caveman")
        body = _json.loads(route.calls[0].request.content)
        assert "Do not summarize or omit facts" in body["messages"][0]["content"]
    finally:
        import os as _os

        _os.environ.pop("MODE", None)


def test_debug_search_requires_admin_and_shows_both_sides():
    client, _ = _client(
        EXA_API_KEY="test-exa",
        PROXY_API_KEY="",
        LLM_BASE_URL="http://llm:8005/v1",
        ADMIN_USER="admin",
        ADMIN_PASS="admin",
    )
    assert client.post("/debug/search", json={"query": "q", "count": 1}).status_code == 401
    with respx.mock:
        respx.post("https://api.exa.ai/search").mock(
            return_value=Response(
                200,
                json={
                    "results": [
                        {
                            "url": "https://example.com/a",
                            "title": "A",
                            "text": "Full raw page text.",
                            "highlights": ["hl1"],
                        }
                    ]
                },
            )
        )
        respx.post("http://llm:8005/v1/chat/completions").mock(
            return_value=Response(200, json={"choices": [{"message": {"content": "LLM summary"}}]})
        )
        r = client.post("/debug/search", json={"query": "q", "count": 1}, headers=_basic())
    assert r.status_code == 200
    item = r.json()["results"][0]
    assert item["snippet"] == "LLM summary"
    assert item["snippet_source"] == "llm"
    assert item["exa_text"] == "Full raw page text."
    assert item["exa_highlights"] == ["hl1"]
    assert item["comparison_summary"] is None  # caveman off: no third pane


def test_debug_search_caveman_adds_comparison_summary():
    client, _ = _client(
        EXA_API_KEY="test-exa",
        PROXY_API_KEY="",
        LLM_BASE_URL="http://llm:8005/v1",
        MODE="original-caveman",
        ADMIN_USER="admin",
        ADMIN_PASS="admin",
    )
    with respx.mock:
        respx.post("https://api.exa.ai/search").mock(
            return_value=Response(
                200,
                json={
                    "results": [
                        {
                            "url": "https://example.com/a",
                            "title": "A",
                            "text": "Full raw page text.",
                            "highlights": ["hl1"],
                        }
                    ]
                },
            )
        )
        llm = respx.post("http://llm:8005/v1/chat/completions").mock(
            side_effect=[
                Response(200, json={"choices": [{"message": {"content": "CAVE"}}]}),
                Response(200, json={"choices": [{"message": {"content": "Normal summary"}}]}),
            ]
        )
        r = client.post("/debug/search", json={"query": "q", "count": 1}, headers=_basic())
    assert r.status_code == 200
    assert r.json()["mode"] == "original-caveman"
    item = r.json()["results"][0]
    assert item["snippet"] == "CAVE"  # what OpenWebUI gets
    assert item["comparison_summary"] == "Normal summary"  # bottom pane
    assert "legacy_caveman_snippet" not in item  # no frozen copies anymore
    assert llm.call_count == 2
    import json as _json

    bodies = [_json.loads(c.request.content) for c in llm.calls]
    assert "Do not summarize or omit facts" in bodies[0]["messages"][0]["content"]
    assert "caveman" not in bodies[1]["messages"][0]["content"].lower()  # normal rules
    assert "caveman" not in bodies[1]["messages"][1]["content"].lower()
    import os as _os

    _os.environ.pop("MODE", None)  # don't leak into other tests


def test_llm_models_autodetect():
    client, _ = _client(
        LLM_BASE_URL="http://llm:8005/v1",
        ADMIN_USER="admin",
        ADMIN_PASS="admin",
    )
    assert client.get("/llm/models").status_code == 401
    with respx.mock:
        respx.get("http://llm:8005/v1/models").mock(
            return_value=Response(
                200,
                json={"data": [{"id": "model-a"}, {"id": "model-b"}, {"nope": 1}]},
            )
        )
        r = client.get("/llm/models", headers=_basic())
    assert r.status_code == 200
    assert r.json()["models"] == ["model-a", "model-b"]


def test_llm_models_backend_down_returns_502():
    client, _ = _client(
        LLM_BASE_URL="http://llm:8005/v1",
        ADMIN_USER="admin",
        ADMIN_PASS="admin",
    )
    with respx.mock:
        respx.get("http://llm:8005/v1/models").mock(
            return_value=Response(500, json={"error": "down"})
        )
        r = client.get("/llm/models", headers=_basic())
    assert r.status_code == 502


def test_llm_slots_lists_pool_and_strips_v1():
    client, _ = _client(
        LLM_BASE_URL="http://llm:8005/v1",
        ADMIN_USER="admin",
        ADMIN_PASS="admin",
    )
    assert client.get("/llm/slots").status_code == 401
    with respx.mock:
        route = respx.get("http://llm:8005/slots").mock(
            return_value=Response(
                200,
                json=[
                    {"id": 0, "is_processing": False},
                    {"id": 1, "is_processing": True},
                    {"id": 2, "is_processing": False},
                ],
            )
        )
        r = client.get("/llm/slots", headers=_basic())
    assert r.status_code == 200
    assert route.called  # /v1 stripped: queried server-native /slots
    assert r.json()["count"] == 3
    assert r.json()["slots"][1] == {"id": 1, "busy": True}


def test_llm_slots_backend_down_returns_502():
    client, _ = _client(
        LLM_BASE_URL="http://llm:8005/v1",
        ADMIN_USER="admin",
        ADMIN_PASS="admin",
    )
    with respx.mock:
        respx.get("http://llm:8005/slots").mock(
            return_value=Response(404, json={"error": "not found"})
        )
        r = client.get("/llm/slots", headers=_basic())
    assert r.status_code == 502


def test_admin_slot_minisection_present():
    client, _ = _client(ADMIN_USER="admin", ADMIN_PASS="admin")
    r = client.get("/admin", headers=_basic())
    assert r.status_code == 200
    assert "LlamaCPP slot IDs" in r.text
    assert "Manage slots" in r.text
    assert "Available slots" in r.text


def _reload_config():
    import importlib

    import app.config as cfg

    importlib.reload(cfg)
    return cfg


def test_config_file_values_load_and_env_wins(tmp_path):
    import json as _json
    import os

    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(_json.dumps({"LLM_MODEL": "file-model", "LLM_TIMEOUT_SEC": 111}))
    os.environ["CONFIG_FILE"] = str(cfg_file)
    os.environ.pop("LLM_MODEL", None)
    os.environ.pop("LLM_TIMEOUT_SEC", None)
    try:
        cfg = _reload_config()
        assert cfg.LLM_MODEL == "file-model"
        assert cfg.LLM_TIMEOUT_SEC == 111
        os.environ["LLM_MODEL"] = "env-model"
        cfg = _reload_config()
        assert cfg.LLM_MODEL == "env-model"  # env beats file
        assert cfg.LLM_TIMEOUT_SEC == 111
    finally:
        os.environ.pop("CONFIG_FILE", None)
        os.environ.pop("LLM_MODEL", None)
        os.environ.pop("LLM_TIMEOUT_SEC", None)
        _reload_config()


def test_update_config_persists_to_config_file(tmp_path):
    import json as _json
    import os

    cfg_file = tmp_path / "config.json"
    os.environ["CONFIG_FILE"] = str(cfg_file)
    os.environ.pop("LLM_MODEL", None)
    try:
        cfg = _reload_config()
        assert cfg.update_config({"LLM_MODEL": "saved-model"}) == ["LLM_MODEL"]
        assert _json.loads(cfg_file.read_text())["LLM_MODEL"] == "saved-model"
        cfg2 = _reload_config()  # survives "restart"
        assert cfg2.LLM_MODEL == "saved-model"
    finally:
        os.environ.pop("CONFIG_FILE", None)
        os.environ.pop("LLM_MODEL", None)
        _reload_config()


def test_legacy_dotenv_imported_when_no_config_file(tmp_path, monkeypatch):
    import os

    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("LLM_MODEL=dotenv-model\n")
    monkeypatch.delenv("CONFIG_FILE", raising=False)  # use default ./config.json
    os.environ.pop("LLM_MODEL", None)
    try:
        cfg = _reload_config()
        assert cfg.LLM_MODEL == "dotenv-model"
    finally:
        os.environ.pop("LLM_MODEL", None)
        _reload_config()


def test_original_mode_returns_raw_text_without_llm():
    import os

    client, _ = _client(
        EXA_API_KEY="test-exa",
        PROXY_API_KEY="",
        LLM_BASE_URL="http://llm:8005/v1",
        MODE="original",
        ADMIN_USER="admin",
        ADMIN_PASS="admin",
    )
    try:
        with respx.mock:
            respx.post("https://api.exa.ai/search").mock(
                return_value=Response(
                    200,
                    json={
                        "results": [
                            {
                                "url": "https://example.com/a",
                                "title": "A",
                                "text": "Raw page text verbatim.",
                                "highlights": ["hl1"],
                            }
                        ]
                    },
                )
            )
            llm = respx.post("http://llm:8005/v1/chat/completions").mock(
                return_value=Response(200, json={"choices": [{"message": {"content": "X"}}]})
            )
            r = client.post("/search", json={"query": "q", "count": 1})
        assert r.status_code == 200
        item = r.json()[0]
        assert item["snippet"] == "Raw page text verbatim."
        assert llm.call_count == 0  # no LLM call in original mode
    finally:
        os.environ.pop("MODE", None)


def test_return_image_urls_appended_as_text():
    import os

    client, _ = _client(
        EXA_API_KEY="test-exa",
        PROXY_API_KEY="",
        LLM_BASE_URL="http://llm:8005/v1",
        MODE="original",
        RETURN_IMAGE_URLS="true",
        ADMIN_USER="admin",
        ADMIN_PASS="admin",
    )
    try:
        with respx.mock:
            respx.post("https://api.exa.ai/search").mock(
                return_value=Response(
                    200,
                    json={
                        "results": [
                            {
                                "url": "https://example.com/a",
                                "title": "A",
                                "text": "Body with ![alt](https://img.example.com/inline.png) pic.",
                                "image": "https://img.example.com/cover.jpg",
                            }
                        ]
                    },
                )
            )
            r = client.post("/debug/search", json={"query": "q", "count": 1}, headers=_basic())
        assert r.status_code == 200
        item = r.json()["results"][0]
        assert "https://img.example.com/cover.jpg" in item["snippet"]
        assert "https://img.example.com/inline.png" in item["snippet"]
        assert item["image_urls"] == [
            "https://img.example.com/cover.jpg",
            "https://img.example.com/inline.png",
        ]
    finally:
        os.environ.pop("MODE", None)
        os.environ.pop("RETURN_IMAGE_URLS", None)


def test_return_image_urls_works_in_summary_mode():
    """Image block is appended post-summary, so it holds in every LLM mode too."""
    import os

    client, _ = _client(
        EXA_API_KEY="test-exa",
        PROXY_API_KEY="",
        LLM_BASE_URL="http://llm:8005/v1",
        MODE="summary",
        RETURN_IMAGE_URLS="true",
        ADMIN_USER="admin",
        ADMIN_PASS="admin",
    )
    try:
        with respx.mock:
            respx.post("https://api.exa.ai/search").mock(
                return_value=Response(
                    200,
                    json={
                        "results": [
                            {
                                "url": "https://example.com/a",
                                "title": "A",
                                "text": "Some body text.",
                                "image": "https://img.example.com/cover.jpg",
                            }
                        ]
                    },
                )
            )
            respx.post("http://llm:8005/v1/chat/completions").mock(
                return_value=Response(200, json={"choices": [{"message": {"content": "Summary."}}]})
            )
            r = client.post("/search", json={"query": "q", "count": 1})
        assert r.status_code == 200
        snippet = r.json()[0]["snippet"]
        assert snippet.startswith("Summary.")
        assert "https://img.example.com/cover.jpg" in snippet
    finally:
        os.environ.pop("MODE", None)
        os.environ.pop("RETURN_IMAGE_URLS", None)


def test_mode_validation_and_legacy_caveman_migration():
    import os

    os.environ.pop("MODE", None)
    os.environ.pop("CAVEMAN_STYLE", None)
    try:
        cfg = _reload_config()
        assert cfg.MODE == "summary"  # default
        cfg.update_config({"MODE": "nonsense"})
        assert cfg.MODE == "summary"  # invalid falls back
        cfg.update_config({"MODE": "Original-Caveman"})
        assert cfg.MODE == "original-caveman"  # normalized
        os.environ["CAVEMAN_STYLE"] = "true"  # legacy flag migrates
        os.environ.pop("MODE", None)
        cfg = _reload_config()
        assert cfg.MODE == "original-caveman"
    finally:
        os.environ.pop("MODE", None)
        os.environ.pop("CAVEMAN_STYLE", None)
        _reload_config()


def test_admin_every_field_has_detailed_tooltip():
    """Each settings field must define a tip strictly longer than its hint."""
    import pathlib
    import re

    html = pathlib.Path("app/admin.html").read_text()
    fields = [
        (m.group(1), m.group(2))
        for m in re.finditer(r"(\w+):\{([^{}]*)\}", html)
        if "label:" in m.group(2)
    ]
    assert len(fields) >= 15, f"expected all settings fields, parsed {len(fields)}"
    for key, body in fields:
        hint = re.search(r"hint:\"(.*?)\"(?=,|$)", body)
        tip = re.search(r"tip:\"(.*?)\"(?=,|$)", body)
        assert hint, f"{key}: missing hint"
        assert tip, f"{key}: missing tip"
        assert len(tip.group(1)) > len(hint.group(1)), f"{key}: tip must be longer than hint"


def _exa_request_contents(env):
    import json as _json
    import os

    os.environ.update(env)
    for var in ("EXA_TEXT_MAX_CHARS",):
        if var not in env:
            os.environ.pop(var, None)
    try:
        import importlib

        import app.config as cfg
        import app.exa as exa_mod

        importlib.reload(cfg)
        importlib.reload(exa_mod)
        with respx.mock:
            route = respx.post("https://api.exa.ai/search").mock(
                return_value=Response(200, json={"results": []})
            )
            import anyio as _anyio

            _anyio.run(exa_mod.exa_search, "q", 1)
        return _json.loads(route.calls[0].request.content)["contents"]
    finally:
        import os as _os

        _os.environ.pop("EXA_TEXT_MAX_CHARS", None)


def test_exa_text_cap_zero_means_unlimited():
    contents = _exa_request_contents({"EXA_API_KEY": "k", "EXA_TEXT_MAX_CHARS": "0"})
    assert contents["text"] is True  # no maxCharacters cap sent
    contents = _exa_request_contents({"EXA_API_KEY": "k", "EXA_TEXT_MAX_CHARS": "500"})
    assert contents["text"] == {"maxCharacters": 500}


@pytest.mark.asyncio
async def test_zero_send_limit_forwards_everything():
    import os

    os.environ["LLM_BASE_URL"] = "http://llm:8005/v1"
    os.environ["SUMMARY_INPUT_MAX_CHARS"] = "0"
    long_text = "x" * 20000
    try:
        import importlib
        import json as _json

        import app.config as cfg
        import app.summarizer as sum_mod

        importlib.reload(cfg)
        importlib.reload(sum_mod)
        with respx.mock:
            route = respx.post("http://llm:8005/v1/chat/completions").mock(
                return_value=Response(200, json={"choices": [{"message": {"content": "s"}}]})
            )
            await sum_mod.summarize_one("q", "t", "http://x", long_text)
        user_msg = _json.loads(route.calls[0].request.content)["messages"][1]["content"]
        assert long_text in user_msg  # nothing truncated
    finally:
        import os as _os

        _os.environ.pop("SUMMARY_INPUT_MAX_CHARS", None)
