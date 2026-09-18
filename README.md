# SSALMP — Simple Search API LLM Summarizer Proxy

Proxy between OpenWebUI (engine external) and a search API (Exa by default)
that returns detailed LLM summaries in snippet instead of raw page text.

Admin web UI included: open /admin on the running container
(e.g. http://<YOUR SERVER IP>:8555/admin) — login admin / admin
(change ADMIN_USER / ADMIN_PASS). Every setting below can be viewed,
tested, and changed there; saves persist across restarts.

```
OpenWebUI                    SSALMP proxy               Search API + LLM
   |  POST /search {query,count}  |                               |
   |----------------------------->|  POST /search                 |
   |                              |------------------------------>|
   |                              |  <-- text + highlights        |
   |                              |  POST /chat/completions       |
   |                              |------------------------------>|
   |                              |  <-- summary                  |
   |  <-- [{link, title, snippet}] |                               |
   |<-----------------------------|                               |
```

## Quick start (docker compose, port 8555)

Prerequisites: Docker, a search API key (Exa: dashboard.exa.ai), and an
OpenAI-compatible LLM endpoint (llama.cpp, vLLM, or SGLang).

```bash
cp .env.example .env
# edit .env: at minimum set EXA_API_KEY and LLM_BASE_URL
# NOTE (containers): LLM_BASE_URL must be reachable FROM the container —
# 127.0.0.1 inside the container means the container itself, so use the
# host LAN IP (e.g. http://10.0.0.187:8003/v1), never localhost.
docker compose up -d --build
curl http://127.0.0.1:8555/healthz
# open http://127.0.0.1:8555/admin (admin/admin) to finish setup in the UI
```

Local run without docker:

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
export EXA_API_KEY=... LLM_BASE_URL=http://127.0.0.1:8005/v1
uvicorn app.main:app --host 0.0.0.0 --port 8555
```

## OpenWebUI setup

`Admin Panel → Settings → Web Search`: Enable, engine `external`,
URL `http://<ssalmp-host>:8555/search`, key = anything (proxy accepts any key).
The admin UI shows the exact URL to paste.

## Endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| POST | `/search` | any (or no) key | OpenWebUI external search |
| POST | `/debug/search` | Basic `ADMIN_USER`/`ADMIN_PASS` | same pipeline + raw search fields per result (`exa_text`, `exa_highlights`, `snippet_source`, `comparison_summary`) for the admin side-by-side view |
| GET | `/llm/models` | Basic `ADMIN_USER`/`ADMIN_PASS` | autodetect model ids from the backend's `/v1/models` (powers the Detect button + suggestions on the LLM model field) |
| GET | `/llm/slots` | Basic `ADMIN_USER`/`ADMIN_PASS` | list llama.cpp slots from server-native `GET /slots` (powers the Detect button on available slots) |
| GET | `/healthz` | none | health + config summary (includes cache stats) |
| GET | `/admin` | Basic `ADMIN_USER`/`ADMIN_PASS` (default `admin`/`admin`) | config web UI |
| GET/POST | `/config` | same Basic | read/update config JSON |
| GET | `/cache/stats` | same Basic | cache size, TTLs, hit/miss counters |
| POST | `/cache/clear` | same Basic | drop all cached entries |

`/search` request: `{"query": "...", "count": 5}` →
response: `[{"link","title","snippet"}]`. Failures return `[]` (OpenWebUI-safe).
Per-result fallback: `LLM summary → search highlights → search text`.
Optional cache (`CACHE_ENABLED=true`): key is `normalized query + count + (MODE, RETURN_IMAGE_URLS)`,
where normalization lowercases and ignores surrounding quotes, extra whitespace, and `. , ; : !`
punctuation.
Fresh hits return `X-Cache: HIT` with no Exa/LLM calls; stale hits revalidate Exa text and return
`X-Cache: REVALIDATED` or regenerate (`X-Cache: MISS`).
Stage 2 (link, `CACHE_LINK_ENABLED=true` by default): after the search API answers, each result URL
(normalized host, no trailing slash/fragment) reuses its stored summary when the stored page text
matches the fresh result exactly — even for a completely different query — saving one LLM call per
hit (`X-Link-Cache: hits/links`). Changed pages regenerate just that link.
Only full successes (no fallback snippets) are cached at either stage. `/debug/search` always bypasses the cache.

## Configuration

Precedence: **built-in defaults < config file < environment variables**.
Every change made in `/admin` is written to the config file immediately and
survives restarts:

| Var | Default | Notes |
|---|---|---|
| `CONFIG_FILE` | `./config.json` | JSON store for all settings below (compose maps it to `/data/config.json` on a named volume) |
| `EXA_API_KEY` | (required) | search provider key (Exa dashboard by default) |
| `EXA_BASE_URL` | `https://api.exa.ai` | search provider base URL |
| `PROXY_API_KEY` | display only | `/search` accepts **any** API key (or none); value shown in admin UI only |
| `ADMIN_USER` / `ADMIN_PASS` | `admin`/`admin` | **Change these**; Basic auth for `/admin`, `/config` |
| `LLM_PROVIDER` | `llamacpp` | `llamacpp` \| `vllm` \| `sglang`. Same chat API for all; only gates backend-specific options (slot pinning) |
| `MODE` | `summary` | `summary` \| `summary-caveman` \| `original` \| `original-caveman`. `summary-caveman` is the same detailed summary in telegraphic style; `original` returns raw text with no LLM call |
| `RETURN_IMAGE_URLS` | `false` | append page image URLs to the snippet body as text, so the downstream LLM can retrieve them |
| `CHUNKED_SUMMARY` | `false` | split pages over `CHUNK_MIN_CHARS` at paragraph/section boundaries, summarize each part with the Mode rules, join outputs verbatim — no final LLM call. For small-context models and very large pages |
| `CHUNK_TARGET_CHARS` | `2600` | target part size when `CHUNKED_SUMMARY` is on; paragraph-boundary based so parts land at/under this |
| `CHUNK_MIN_CHARS` | `5000` | trigger: only pages longer than this are chunked (and only when enabled); shorter pages summarize whole |
| `CAVEMAN_STYLE` | legacy | `true` behaves like `MODE=original-caveman` when `MODE` is unset |
| `LLM_BASE_URL` | `http://10.0.0.187:8005/v1` | OpenAI-compatible base URL of your backend |
| `LLM_MODEL` | `Muse-Glimmer-30B` | must match a backend model id (use Detect in `/admin`) |
| `LLAMACPP_USE_SLOTS` | `false` | llama.cpp only: rotate each job across the slot pool (`id_slot` 0..N-1) instead of auto-assign; a job on a busy slot waits its turn |
| `LLAMACPP_SLOT_COUNT` | `1` | pool size for rotation: N slots = ids 0..N-1. Click Detect in `/admin` to read the live count from `GET /slots` (legacy `LLAMACPP_SLOT_ID=N` migrates to a pool of N+1) |
| `LLM_MAX_TOKENS` | `0` | cap per summary; `0`/negative omits it (server default = unlimited). Keep ≥1500 when set on reasoning backends: reasoning tokens come out of this budget; small values return empty content |
| `LLM_TIMEOUT_SEC` | `300` | per-summary timeout; large reasoning models are slow |
| `EXA_TEXT_MAX_CHARS` | `0` | chars fetched per result; `0` = unlimited (omit the cap) |
| `MAX_CONCURRENT_SUMMARIES` | `1` | global single-flight: only 1 summary runs at a time, rest queue (raise only if the inference engine handles parallel jobs) |
| `REQUEST_TIMEOUT_SEC` | `1200` | whole-search timeout incl. queue wait |
| `SUMMARY_INPUT_MAX_CHARS` | `0` | chars forwarded per summary; `0` = unlimited (forward everything fetched) |
| `SUMMARY_MIN_CHARS` | `10000` | pages shorter than this skip the local LLM and pass through verbatim; `0` = summarize everything |
| `CACHE_ENABLED` | `false` | stage-1 query cache: repeat query + count with same `MODE` + `RETURN_IMAGE_URLS` returns cached summary |
| `CACHE_LINK_ENABLED` | `true` | stage-2 link cache (needs `CACHE_ENABLED`): per-URL summaries reused across queries when page text matches; shares TTL, size cap and file with stage 1 |
| `CACHE_FRESH_SEC` | `86400` | fresh window (24h): hits return directly with zero Exa/LLM calls |
| `CACHE_TTL_SEC` | `172800` | outer TTL (48h): past the fresh window the proxy re-runs Exa and reuses the cache only if fresh results match the stored text; older entries expire |
| `CACHE_MAX_ENTRIES` | `200` | LRU cap on cached entries (queries + link snippets share the pool; memory + database) |
| `CACHE_DB_FILE` | `./cache.db` (`/data/cache.db` in compose) | SQLite backing store (WAL mode, row-level writes); survives restarts via the named volume; inspect with `sqlite3 /data/cache.db "SELECT key,kind,datetime(created_at,'unixepoch') FROM entries;"` |
| `CACHE_FILE` | `./cache.json` (legacy) | previous JSON store: imported once into `CACHE_DB_FILE` on first start when the database is empty, then ignored; safe to delete |

On first start with no `config.json`, a legacy `.env` file in the working
directory is read once as the starting point (for upgrades from earlier
versions).

## Tests

```bash
pip install -r requirements.txt
pytest -q
```
