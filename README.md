# SSALMP — Simple Search API LLM Summarizer Proxy

Proxy between **OpenWebUI** (engine `external`) and a **search API** (Exa by
default) that returns **detailed LLM summaries** in `snippet` instead of raw
page text.

```
OpenWebUI --POST /search {query,count}--> proxy --POST /search--> Search API
  <--[{link,title,snippet=summary}]--  <--text+highlights--  Search API
                                          --chat/completions--> LLM backend
                                          <--summary-----------
```

## Endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| POST | `/search` | any (or no) key | OpenWebUI external search |
| POST | `/debug/search` | Basic `ADMIN_USER`/`ADMIN_PASS` | same pipeline + raw search fields per result (`exa_text`, `exa_highlights`, `snippet_source`, `comparison_summary`) for the admin side-by-side view |
| GET | `/llm/models` | Basic `ADMIN_USER`/`ADMIN_PASS` | autodetect model ids from the backend's `/v1/models` (powers the Detect button + suggestions on the LLM model field) |
| GET | `/healthz` | none | health + config summary |
| GET | `/admin` | Basic `ADMIN_USER`/`ADMIN_PASS` (default `admin`/`admin`) | config web UI |
| GET/POST | `/config` | same Basic | read/update config JSON |

`/search` request: `{"query": "...", "count": 5}` →
response: `[{"link","title","snippet"}]`. Failures return `[]` (OpenWebUI-safe).
Per-result fallback: `LLM summary → search highlights → search text`.

## Quick start (docker compose, port 8555)

Prerequisites: Docker, a search API key (Exa: dashboard.exa.ai), and an
OpenAI-compatible LLM endpoint (llama.cpp, vLLM, or SGLang).

```bash
cp .env.example .env
# edit .env: at minimum set EXA_API_KEY and LLM_BASE_URL
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
| `MODE` | `summary` | `summary` \| `summary-caveman` \| `original` \| `original-caveman`. `original` returns raw text with no LLM call; `*-caveman` uses telegraphic style |
| `RETURN_IMAGE_URLS` | `false` | append page image URLs to the snippet body as text, so the downstream LLM can retrieve them |
| `CAVEMAN_STYLE` | legacy | `true` behaves like `MODE=original-caveman` when `MODE` is unset |
| `LLM_BASE_URL` | `http://10.0.0.187:8005/v1` | OpenAI-compatible base URL of your backend |
| `LLM_MODEL` | `Muse-Glimmer-30B` | must match a backend model id (use Detect in `/admin`) |
| `LLAMACPP_USE_SLOTS` | `false` | llama.cpp only: send `id_slot` to pin each job to one slot instead of any idle slot |
| `LLAMACPP_SLOT_ID` | `1` | slot id, 1 or above (server slots are 0-based and wrap) |
| `LLM_MAX_TOKENS` | `0` | cap per summary; `0`/negative omits it (server default = unlimited). Keep ≥1500 when set on reasoning backends: reasoning tokens come out of this budget; small values return empty content |
| `LLM_TIMEOUT_SEC` | `300` | per-summary timeout; large reasoning models are slow |
| `EXA_TEXT_MAX_CHARS` | `8000` | text chars requested per result from the search API |
| `MAX_CONCURRENT_SUMMARIES` | `1` | global single-flight: only 1 summary runs at a time, rest queue (raise only if the inference engine handles parallel jobs) |
| `REQUEST_TIMEOUT_SEC` | `1200` | whole-search timeout incl. queue wait |
| `SUMMARY_INPUT_MAX_CHARS` | `12000` | page text chars sent to the LLM backend |

On first start with no `config.json`, a legacy `.env` file in the working
directory is read once as the starting point (for upgrades from earlier
versions).

## OpenWebUI setup

`Admin Panel → Settings → Web Search`: Enable, engine `external`,
URL `http://<ssalmp-host>:8555/search`, key = anything (proxy accepts any key).
The admin UI shows the exact URL to paste.

## Example summarizer backend (llama.cpp)

```bash
export CUDA_VISIBLE_DEVICES=2,3
nohup /home/user/llama.cpp/build/bin/llama-server \
 -m /home/user/models/Muse-Glimmer-30B/Muse-Glimmer-30B-UD-Q4_K_XL.gguf \
 --port 8005 --host 0.0.0.0 -a Muse-Glimmer-30B \
 --no-mmap --threads 8 --jinja \
 --flash-attn on -kvu -np 4 -sm layer -ngl 99 \
 -md /home/user/models/Muse-Glimmer-30B/dflash-kquant.gguf -ngld 99 \
 --spec-type draft-dflash --spec-draft-n-max 10 \
 --temp 1.0 --top-p 0.95 --top-k 64 \
 --reasoning on --reasoning-preserve \
 -mm /home/user/models/Muse-Glimmer-30B/mmproj-Muse-Glimmer-30B-BF16.gguf \
 -lv 4 &
curl http://127.0.0.1:8005/health
curl http://127.0.0.1:8005/v1/models
```

## Tests

```bash
pip install -r requirements.txt
pytest -q
```
