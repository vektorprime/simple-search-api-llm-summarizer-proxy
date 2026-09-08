"""Layered configuration: built-in defaults < config file < environment.

The config file (JSON, default ./config.json, override with CONFIG_FILE)
survives restarts — the admin UI writes every change there. Environment
variables always win, so docker -e / compose values still override.
If no JSON file exists yet, a legacy .env file in the working directory
is read once as the starting point (not written back).
"""
from __future__ import annotations

import json
import os

CONFIG_FILE: str = os.environ.get("CONFIG_FILE", "./config.json")


def _load_file_values() -> dict:
    path = CONFIG_FILE
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}
    if os.path.abspath(path) != os.path.abspath("./config.json"):
        return {}  # explicit CONFIG_FILE that doesn't exist: don't sniff .env
    # one-time legacy import: plain KEY=VALUE .env in the working directory
    # one-time legacy import: plain KEY=VALUE .env in the working directory
    dotenv = os.path.join(os.getcwd(), ".env")
    values: dict[str, str] = {}
    if os.path.exists(dotenv):
        try:
            with open(dotenv) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        values[k.strip()] = v.strip()
        except OSError:
            pass
    return values


_FILE_VALUES: dict = _load_file_values()


def _raw(name: str) -> str | None:
    if name in os.environ:
        return os.environ[name]
    v = _FILE_VALUES.get(name)
    return str(v) if v is not None else None


def _get(name: str, default: str = "") -> str:
    v = _raw(name)
    return v if v is not None else default


def _get_int(name: str, default: int) -> int:
    v = _raw(name)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    v = _raw(name)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _get_bool(name: str, default: bool) -> bool:
    v = _raw(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


# --- Required / auth ---
EXA_API_KEY: str = _get("EXA_API_KEY", "")
PROXY_API_KEY: str = _get("PROXY_API_KEY", "")  # display only; /search accepts any key

# --- LLM (OpenAI-compatible backend, e.g. llama.cpp) ---
LLM_PROVIDER: str = _get("LLM_PROVIDER", "llamacpp").strip().lower() or "llamacpp"
LLM_BASE_URL: str = _get("LLM_BASE_URL", "http://10.0.0.187:8005/v1")
LLM_MODEL: str = _get("LLM_MODEL", "Muse-Glimmer-30B")
LLM_API_KEY: str = _get("LLM_API_KEY", "sk-no-key-required")
LLM_TIMEOUT_SEC: int = _get_int("LLM_TIMEOUT_SEC", 300)
LLM_MAX_TOKENS: int = _get_int("LLM_MAX_TOKENS", 0)
# llama.cpp slot management: only used when provider == llamacpp and enabled.
# Instead of pinning one slot, the proxy rotates each job across the first
# LLAMACPP_SLOT_COUNT slots (0-based ids 0..N-1; out-of-range wraps server-side).
# A job pinned to a busy slot simply waits its turn (server defers, no error).
LLAMACPP_USE_SLOTS: bool = _get_bool("LLAMACPP_USE_SLOTS", False)


def _resolve_slot_count() -> int:
    raw = _raw("LLAMACPP_SLOT_COUNT")
    if raw is not None:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    legacy = _raw("LLAMACPP_SLOT_ID")  # pre-rotation single slot id
    if legacy is not None:
        try:
            # pinned id N implies a pool of at least N+1 slots (0..N)
            return max(1, int(legacy) + 1)
        except ValueError:
            pass
    return 1


LLAMACPP_SLOT_COUNT: int = _resolve_slot_count()
# Caveman style: ultra-terse summaries to save tokens.
# (Superseded by MODE below, but still read for backwards compatibility.)
_LEGACY_CAVEMAN: bool = _get_bool("CAVEMAN_STYLE", False)

# --- Operation mode ---
# summary: detailed LLM summary (default)
# summary-caveman: LLM summary in telegraphic caveman style
# original: raw search text, unchanged, no LLM call
# original-caveman: full telegraphic rewrite via LLM, nothing omitted
_VALID_MODES = ("summary", "summary-caveman", "original", "original-caveman")


def _resolve_mode() -> str:
    raw = _raw("MODE")
    if raw is None:
        return "original-caveman" if _LEGACY_CAVEMAN else "summary"
    mode = raw.strip().lower()
    return mode if mode in _VALID_MODES else "summary"


MODE: str = _resolve_mode()
# Append page image URLs to the snippet body as text (downstream LLM can use them).
RETURN_IMAGE_URLS: bool = _get_bool("RETURN_IMAGE_URLS", False)

# --- Exa ---
EXA_BASE_URL: str = _get("EXA_BASE_URL", "https://api.exa.ai")
EXA_TIMEOUT_SEC: int = _get_int("EXA_TIMEOUT_SEC", 20)
# 0 = unlimited (omit maxCharacters; Exa returns its default text).
EXA_TEXT_MAX_CHARS: int = _get_int("EXA_TEXT_MAX_CHARS", 0)

# --- Proxy behaviour ---
# NOTE: llama.cpp on 2x3080 effectively runs 1 summarization at a time;
# the proxy enforces this globally (across requests) and queues the rest.
PORT: int = _get_int("PORT", 8555)
MAX_CONCURRENT_SUMMARIES: int = _get_int("MAX_CONCURRENT_SUMMARIES", 1)
# 0 = unlimited (forward everything fetched).
SUMMARY_INPUT_MAX_CHARS: int = _get_int("SUMMARY_INPUT_MAX_CHARS", 0)
REQUEST_TIMEOUT_SEC: int = _get_int("REQUEST_TIMEOUT_SEC", 1200)

# --- Admin UI login (HTTP Basic Auth for /admin and /config) ---
ADMIN_USER: str = _get("ADMIN_USER", "admin")
ADMIN_PASS: str = _get("ADMIN_PASS", "admin")

# --- Runtime admin UI support (in-memory overrides + CONFIG_FILE persist) ---
_EDITABLE_STR = (
    "EXA_API_KEY",
    "PROXY_API_KEY",
    "LLM_PROVIDER",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "LLM_API_KEY",
    "EXA_BASE_URL",
    "MODE",
)
_EDITABLE_INT = (
    "LLM_TIMEOUT_SEC",
    "LLM_MAX_TOKENS",
    "LLAMACPP_SLOT_COUNT",
    "EXA_TIMEOUT_SEC",
    "EXA_TEXT_MAX_CHARS",
    "MAX_CONCURRENT_SUMMARIES",
    "SUMMARY_INPUT_MAX_CHARS",
    "REQUEST_TIMEOUT_SEC",
)
_EDITABLE_FLOAT: tuple[str, ...] = ()
_EDITABLE_BOOL = ("LLAMACPP_USE_SLOTS", "RETURN_IMAGE_URLS")

EDITABLE_FIELDS: tuple[str, ...] = _EDITABLE_STR + _EDITABLE_INT + _EDITABLE_FLOAT + _EDITABLE_BOOL
_SECRET_FIELDS = {"EXA_API_KEY", "PROXY_API_KEY", "LLM_API_KEY"}


def _mask(name: str, value: str) -> str:
    if name in _SECRET_FIELDS and value:
        if len(value) <= 8:
            return "****"
        return value[:4] + "****" + value[-4:]
    return value


def public_config() -> dict:
    """Current config for the admin UI (secrets masked)."""
    import sys

    mod = sys.modules[__name__]
    out: dict = {}
    for name in EDITABLE_FIELDS:
        val = getattr(mod, name)
        out[name] = _mask(name, str(val)) if isinstance(val, str) else val
    out["PORT"] = PORT
    return out


def update_config(updates: dict) -> list[str]:
    """Apply in-memory updates; returns list of applied field names.

    Empty-string values for secret fields mean 'leave unchanged'
    (so the UI can submit masked placeholders safely).
    """
    import sys

    mod = sys.modules[__name__]
    applied: list[str] = []
    for name in EDITABLE_FIELDS:
        if name not in updates:
            continue
        raw = updates[name]
        if raw is None:
            continue
        if isinstance(raw, str) and not raw.strip() and name in _SECRET_FIELDS:
            continue  # leave secret unchanged
        try:
            if name in _EDITABLE_INT:
                setattr(mod, name, int(raw))
            elif name in _EDITABLE_FLOAT:
                setattr(mod, name, float(raw))
            elif name in _EDITABLE_BOOL:
                if isinstance(raw, bool):
                    setattr(mod, name, raw)
                else:
                    setattr(mod, name, str(raw).strip().lower() in ("1", "true", "yes", "on"))
            elif name == "LLM_PROVIDER":
                setattr(mod, name, str(raw).strip().lower() or "llamacpp")
            elif name == "MODE":
                mode = str(raw).strip().lower()
                setattr(mod, name, mode if mode in _VALID_MODES else "summary")
            else:
                setattr(mod, name, str(raw).strip())
            applied.append(name)
        except (ValueError, TypeError):
            continue
    try:
        save_config_file(applied)
    except Exception:
        pass
    return applied


def save_config_file(names: list[str]) -> None:
    """Best-effort: persist applied keys to CONFIG_FILE (JSON) so restarts keep them.

    Merges with existing file content; unknown keys are preserved.
    """
    import sys

    if not names:
        return
    if os.environ.get("PYTEST_CURRENT_TEST") and not os.environ.get("CONFIG_FILE"):
        return  # never let the test suite write ./config.json in the repo
    mod = sys.modules[__name__]
    path = CONFIG_FILE
    existing: dict = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, dict):
                existing = data
        except (OSError, ValueError):
            existing = {}
    for name in names:
        existing[name] = getattr(mod, name)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(existing, f, indent=2)
        f.write("\n")
    os.replace(tmp, path)
