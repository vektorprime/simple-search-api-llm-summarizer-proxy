"""Optional exact-match (case-insensitive) query cache with disk persistence.

Semantics (per requirements):
- Key = normalized query + count + fingerprint. Normalization: strip,
  unicode-quote folding, lower, separator folding (`. , ; : !` -> spaces),
  internal-whitespace collapse, surrounding quote/bracket stripping (so
  `"Paris!"`, `“Paris”` and `  paris  ` all hit).
- Fingerprint = MODE + RETURN_IMAGE_URLS only.
- Tiered TTL:
  - age <= CACHE_FRESH_SEC (default 86400, 24h): fresh HIT, return payload
    immediately with zero Exa/LLM calls.
  - FRESH < age <= CACHE_TTL_SEC (default 172800, 48h): stale window.
    Caller must run a fresh exa_search, canonicalize it, and compare to the
    stored exa_items. If equal -> revalidated HIT (return cached payload,
    refresh timestamp). If different -> MISS (regenerate + put).
  - age > CACHE_TTL_SEC: expired, evicted, MISS.
- Entry stores BOTH the /search payload AND the canonical Exa items used to
  build it, so stale revalidation can compare old vs new search text.
- Stage 2 (link cache): per-URL summary entries keyed by normalized URL +
  fingerprint, WITHOUT the query — so different queries returning the same
  page share one summary. On lookup the stored page text must match the
  fresh Exa text exactly, else that link regenerates. Queries and links
  share one LRU pool, TTL and backing file (entries tagged by key prefix).
- Disk persistence: SQLite file at config.CACHE_DB_FILE (defaults to
  cache.db next to CONFIG_FILE, so /data/cache.db in docker). In-memory LRU
  dict stays the fast path; every mutation is written through as a single
  row (no whole-file rewrites). A legacy cache.json at config.CACHE_FILE,
  if present, is imported once on first start (keys recomputed) and then
  left untouched.

Only successful results should be cached — the caller (app/main.py) decides
what "success" means and calls put() accordingly. This module never calls
Exa or the LLM itself.
"""
from __future__ import annotations

import collections
import hashlib
import json
import logging
import os
import sqlite3
import threading
import time

from app import config as _config  # local import to read live values (tests reload)

log = logging.getLogger("search-proxy")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
  key TEXT PRIMARY KEY,
  kind TEXT NOT NULL,
  created_at REAL NOT NULL,
  data TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_entries_created ON entries(created_at);
"""

_QUERY_PREFIX = "q:"
_LINK_PREFIX = "u:"

_lock = threading.RLock()
# OrderedDict[key -> entry] in LRU order (oldest first). Loaded lazily.
_store: collections.OrderedDict[str, dict] | None = None

# Simple hit/miss counters (in-memory only, reset on restart).
_hits_fresh = 0
_hits_revalidated = 0
_misses = 0
_link_hits = 0
_link_misses = 0


_QUOTE_FOLD = {
    "“": '"',
    "”": '"',
    "„": '"',
    "«": '"',
    "»": '"',
    "‘": "'",
    "’": "'",
    "‚": "'",
    "‹": "'",
    "›": "'",
}

_SURROUNDING_STRIP = "\"'“”‘’«»()[]{}"

# Punctuation treated as separators (folded to spaces, so `study: med`
# and `study med` share a key). Kept to the exact user-specified set.
_SEPARATOR_CHARS = ".,;:!"


def normalize_query(query: str) -> str:
    """Forgiving case-insensitive normalization.

    Steps: trim -> fold unicode quotes to straight -> lower -> fold the
    separator chars (`. , ; : !`) to spaces -> collapse all internal
    whitespace runs to single spaces -> strip surrounding quotes/brackets.
    """
    q = (query or "").strip()
    if not q:
        return ""
    for src, dst in _QUOTE_FOLD.items():
        if src in q:
            q = q.replace(src, dst)
    q = q.lower()
    for ch in _SEPARATOR_CHARS:
        if ch in q:
            q = q.replace(ch, " ")
    q = " ".join(q.split())
    q = q.strip(_SURROUNDING_STRIP).strip()
    return q


def fingerprint() -> dict:
    """Settings fingerprint: MODE + RETURN_IMAGE_URLS only (per spec)."""
    return {
        "MODE": _config.MODE,
        "RETURN_IMAGE_URLS": bool(_config.RETURN_IMAGE_URLS),
    }


def cache_key(norm_query: str, count: int, fp: dict) -> str:
    """Stable stage-1 key over normalized query + count + fingerprint."""
    raw = json.dumps(
        {"q": norm_query, "count": int(count), "fp": fp},
        sort_keys=True,
        separators=(",", ":"),
    )
    return _QUERY_PREFIX + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def normalize_url(url: str) -> str:
    """Normalize a result URL for stage-2 link keys.

    Lowercases scheme+host (case-insensitive per RFC), preserves path case
    and query string, drops fragments (client-side only) and trailing
    slashes, so `https://Example.com/a/` and `https://example.com/a` hit.
    Unparseable input falls back to strip + trailing-slash removal.
    """
    u = (url or "").strip()
    if not u:
        return ""
    try:
        from urllib.parse import urlsplit, urlunsplit

        p = urlsplit(u)
        if not p.scheme and not p.netloc:
            return u.rstrip("/")
        netloc = (p.netloc or "").lower()
        path = (p.path or "").rstrip("/")
        return urlunsplit((p.scheme.lower(), netloc, path, p.query, ""))
    except Exception:
        return u.rstrip("/")


def link_key(url_norm: str, fp: dict) -> str:
    """Stable stage-2 key over normalized URL + fingerprint (no query)."""
    raw = json.dumps(
        {"u": url_norm, "fp": fp},
        sort_keys=True,
        separators=(",", ":"),
    )
    return _LINK_PREFIX + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def canonical_exa(items: list[dict]) -> list[dict]:
    """Reduce raw Exa result dicts to the comparable subset, preserving order.

    Only fields that feed the snippet pipeline are kept: url, title, text,
    highlights, image. Extra provider fields are ignored so harmless metadata
    drift does not invalidate the cache.
    """
    out: list[dict] = []
    for it in items or []:
        if not isinstance(it, dict):
            continue
        hl = it.get("highlights")
        out.append(
            {
                "url": it.get("url") or it.get("link") or "",
                "title": it.get("title"),
                "text": it.get("text") or "",
                "highlights": list(hl) if isinstance(hl, list) else [],
                "image": it.get("image"),
            }
        )
    return out


def exa_equal(a: list[dict] | None, b: list[dict] | None) -> bool:
    """Order-sensitive exact equality of canonical Exa lists."""
    return (a or []) == (b or [])


def _db_file() -> str:
    return getattr(_config, "CACHE_DB_FILE", "") or ""


def _legacy_json_file() -> str:
    return getattr(_config, "CACHE_FILE", "") or ""


def _persistence_enabled() -> bool:
    # Never let the test suite touch real files in the repo unless the test
    # explicitly points the cache at a tmp path.
    if os.environ.get("PYTEST_CURRENT_TEST") and not (
        os.environ.get("CACHE_DB_FILE") or os.environ.get("CACHE_FILE")
    ):
        return False
    return True


_conn: sqlite3.Connection | None = None
_conn_path: str | None = None
_db_warned = False


def _warn_once(msg: str, *args) -> None:
    global _db_warned
    if not _db_warned:
        _db_warned = True
        log.warning(msg, *args)


def _close_locked() -> None:
    global _conn, _conn_path
    if _conn is not None:
        try:
            _conn.close()
        except sqlite3.Error:
            pass
        _conn = None
        _conn_path = None


def _connect_locked() -> sqlite3.Connection | None:
    """Open (or reuse) the SQLite cache DB. Must be called with _lock held."""
    global _conn, _conn_path
    if not _persistence_enabled():
        return None
    path = _db_file()
    if not path:
        return None
    if _conn is not None and _conn_path == path:
        return _conn
    _close_locked()
    try:
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        conn = sqlite3.connect(path, timeout=5.0, check_same_thread=False,
                               isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.executescript(_SCHEMA)
    except (OSError, sqlite3.Error) as e:
        _warn_once("cache db unavailable at %s (%s); running memory-only", path, e)
        return None
    _conn = conn
    _conn_path = path
    return conn


def _migrate_json_once_locked(conn: sqlite3.Connection) -> int:
    """One-time import of the legacy cache.json. Keys are recomputed with the
    current key scheme; unparseable rows are skipped. The JSON file itself is
    left untouched. Returns rows imported."""
    try:
        if conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]:
            return 0  # DB already populated: nothing to migrate
    except sqlite3.Error:
        return 0
    src = _legacy_json_file()
    if not src or not os.path.exists(src):
        return 0
    try:
        with open(src, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return 0
    entries = data.get("entries") if isinstance(data, dict) else None
    if not isinstance(entries, dict):
        return 0
    rows: list[tuple[str, str, float, str]] = []
    for e in entries.values():
        if not isinstance(e, dict):
            continue
        try:
            created = float(e.get("created_at", 0))
        except (TypeError, ValueError):
            continue
        fp = e.get("fingerprint")
        if not isinstance(fp, dict):
            continue
        if isinstance(e.get("payload"), list) and isinstance(e.get("exa_items"), list):
            kind = "query"
            try:
                key = cache_key(e.get("query_norm") or "", int(e.get("count", 0)), fp)
            except (TypeError, ValueError):
                continue
        elif isinstance(e.get("snippet"), str) and isinstance(e.get("exa"), dict):
            kind = "link"
            key = link_key(normalize_url(e.get("url_norm") or ""), fp)
        else:
            continue
        e = dict(e)
        e["kind"] = kind
        rows.append((key, kind, created, json.dumps(e)))
    if not rows:
        return 0
    try:
        conn.executemany(
            "INSERT OR IGNORE INTO entries(key, kind, created_at, data) VALUES (?,?,?,?)",
            rows,
        )
    except sqlite3.Error as ex:
        _warn_once("cache json migration failed (%s)", ex)
        return 0
    log.info("cache migrated %d entries from %s", len(rows), src)
    return len(rows)


def _prune_locked(conn: sqlite3.Connection, store: collections.OrderedDict[str, dict],
                  ttl: float, max_n: int) -> None:
    """Drop expired/over-cap rows from both memory and DB. VACUUMs on change."""
    now = time.time()
    dead = [k for k, e in store.items() if now - float(e.get("created_at", 0)) > ttl]
    while len(store) - len(dead) > max_n:
        for k in store:
            if k not in dead:
                dead.append(k)
                break
    if not dead:
        return
    for k in dead:
        store.pop(k, None)
    try:
        conn.executemany("DELETE FROM entries WHERE key=?", [(k,) for k in dead])
        conn.execute("VACUUM")
    except sqlite3.Error:
        pass


def _load_from_db() -> collections.OrderedDict[str, dict]:
    store: collections.OrderedDict[str, dict] = collections.OrderedDict()
    with _lock:
        conn = _connect_locked()
        if conn is None:
            return store
        try:
            _migrate_json_once_locked(conn)
            rows = conn.execute(
                "SELECT key, kind, created_at, data FROM entries ORDER BY rowid"
            ).fetchall()
        except sqlite3.Error:
            return store
        for key, kind, _created, blob in rows:
            try:
                e = json.loads(blob)
            except ValueError:
                continue
            if not isinstance(e, dict):
                continue
            e["kind"] = kind
            store[key] = e
        _prune_locked(conn, store, max(1, int(_config.CACHE_TTL_SEC)),
                      max(1, int(_config.CACHE_MAX_ENTRIES)))
        return store


def _ensure_loaded() -> collections.OrderedDict[str, dict]:
    global _store
    if _store is None:
        _store = _load_from_db()
    return _store


def _write_row_locked(key: str, entry: dict) -> None:
    conn = _connect_locked()
    if conn is None:
        return
    try:
        conn.execute(
            "INSERT OR REPLACE INTO entries(key, kind, created_at, data) VALUES (?,?,?,?)",
            (key, entry.get("kind", "query"), float(entry.get("created_at", 0)),
             json.dumps(entry)),
        )
    except sqlite3.Error as e:
        _warn_once("cache db write failed (%s); running memory-only", e)


def _delete_row_locked(key: str) -> None:
    conn = _connect_locked()
    if conn is None:
        return
    try:
        conn.execute("DELETE FROM entries WHERE key=?", (key,))
    except sqlite3.Error:
        pass


def _enforce_max_locked(store: collections.OrderedDict[str, dict]) -> None:
    max_n = max(1, int(_config.CACHE_MAX_ENTRIES))
    while len(store) > max_n:
        old, _ = store.popitem(last=False)
        _delete_row_locked(old)


def get_fresh(key: str) -> list[dict] | None:
    """Return cached payload on fresh HIT, else None.

    Moves the entry to MRU position on hit. Expired entries are evicted.
    Stale-window entries return None here (caller uses get_stale + compare).
    """
    global _hits_fresh, _misses
    with _lock:
        store = _ensure_loaded()
        e = store.get(key)
        if e is None:
            _misses += 1
            return None
        try:
            created = float(e.get("created_at", 0))
        except (TypeError, ValueError):
            store.pop(key, None)
            _misses += 1
            return None
        now = time.time()
        fresh_sec = max(1, int(_config.CACHE_FRESH_SEC))
        ttl_sec = max(1, int(_config.CACHE_TTL_SEC))
        age = now - created
        if age <= fresh_sec:
            store.move_to_end(key)
            _hits_fresh += 1
            payload = e.get("payload")
            return list(payload) if isinstance(payload, list) else None
        if age > ttl_sec:
            store.pop(key, None)
            _delete_row_locked(key)
            _misses += 1
            return None
        # Stale window: not a fresh hit; caller must revalidate via get_stale.
        _misses += 1
        return None


def get_stale(key: str) -> dict | None:
    """Return the full entry iff it is in the stale revalidation window.

    Fresh hits and expired/missing keys return None. Does NOT move LRU
    (refresh() does that on successful revalidation).
    """
    with _lock:
        store = _ensure_loaded()
        e = store.get(key)
        if e is None:
            return None
        try:
            created = float(e.get("created_at", 0))
        except (TypeError, ValueError):
            return None
        now = time.time()
        fresh_sec = max(1, int(_config.CACHE_FRESH_SEC))
        ttl_sec = max(1, int(_config.CACHE_TTL_SEC))
        age = now - created
        if fresh_sec < age <= ttl_sec:
            return e
        return None


def put(
    key: str,
    norm_query: str,
    count: int,
    fp: dict,
    payload: list[dict],
    exa_items: list[dict],
) -> None:
    """Store payload + canonical Exa items. Evicts LRU oldest when full."""
    with _lock:
        store = _ensure_loaded()
        store[key] = {
            "kind": "query",
            "query_norm": norm_query,
            "count": int(count),
            "fingerprint": dict(fp),
            "created_at": time.time(),
            "payload": list(payload),
            "exa_items": list(exa_items or []),
        }
        store.move_to_end(key)
        _write_row_locked(key, store[key])
        _enforce_max_locked(store)


def _link_content_equal(stored: dict | None, fresh: dict | None) -> bool:
    """Exact content compare with URL fields normalized first.

    `https://example.com/a` and `https://example.com/a/` are the same page;
    everything else (title/text/highlights/image) must match byte-for-byte.
    """
    if not isinstance(stored, dict) or not isinstance(fresh, dict):
        return (stored or {}) == (fresh or {})
    s = dict(stored)
    f = dict(fresh)
    s["url"] = normalize_url(s.get("url") or s.get("link") or "")
    f["url"] = normalize_url(f.get("url") or f.get("link") or "")
    return s == f


def get_link_hit(key: str, fresh_one: dict) -> tuple[str | None, str | None]:
    """Stage-2 lookup: return (snippet, source) on HIT, (None, None) on MISS.

    Hits require: entry present, kind == link, unexpired (age <=
    CACHE_TTL_SEC), and stored page content exactly equal to the fresh
    canonical item. Hits move to MRU; expired entries are evicted.
    Mismatched-content entries are kept (put_link overwrites on regen).
    """
    global _link_hits, _link_misses
    with _lock:
        store = _ensure_loaded()
        e = store.get(key)
        if e is None or e.get("kind") != "link":
            _link_misses += 1
            return None, None
        try:
            created = float(e.get("created_at", 0))
        except (TypeError, ValueError):
            store.pop(key, None)
            _delete_row_locked(key)
            _link_misses += 1
            return None, None
        if time.time() - created > max(1, int(_config.CACHE_TTL_SEC)):
            store.pop(key, None)
            _delete_row_locked(key)
            _link_misses += 1
            return None, None
        if not _link_content_equal(e.get("exa"), fresh_one):
            _link_misses += 1
            return None, None
        store.move_to_end(key)
        _link_hits += 1
        return e.get("snippet"), e.get("source")


def put_link(
    key: str,
    url_norm: str,
    fp: dict,
    snippet: str,
    source: str,
    exa_one: dict,
) -> None:
    """Store one link snippet + its canonical page content. LRU-evicted."""
    with _lock:
        store = _ensure_loaded()
        store[key] = {
            "kind": "link",
            "url_norm": url_norm,
            "fingerprint": dict(fp),
            "created_at": time.time(),
            "snippet": snippet,
            "source": source,
            "exa": dict(exa_one or {}),
        }
        store.move_to_end(key)
        _write_row_locked(key, store[key])
        _enforce_max_locked(store)


def refresh(key: str) -> None:
    """Bump a revalidated entry to now + MRU and persist."""
    with _lock:
        store = _ensure_loaded()
        e = store.get(key)
        if e is None:
            return
        e["created_at"] = time.time()
        store.move_to_end(key)
        _write_row_locked(key, e)
    global _hits_revalidated
    with _lock:
        _hits_revalidated += 1


def _clear_all_locked() -> None:
    conn = _connect_locked()
    if conn is None:
        return
    try:
        conn.execute("DELETE FROM entries")
    except sqlite3.Error:
        pass


def clear() -> int:
    """Remove all entries (memory + disk). Returns number removed."""
    with _lock:
        store = _ensure_loaded()
        n = len(store)
        store.clear()
        _clear_all_locked()
        return n


def stats() -> dict:
    """JSON-serializable stats for /cache/stats and /healthz."""
    with _lock:
        store = _ensure_loaded()
        kinds = [e.get("kind", "query") for e in store.values()]
        return {
            "enabled": bool(_config.CACHE_ENABLED),
            "link_enabled": bool(getattr(_config, "CACHE_LINK_ENABLED", True)),
            "size": len(store),
            "query_entries": sum(1 for k in kinds if k != "link"),
            "link_entries": sum(1 for k in kinds if k == "link"),
            "max_entries": int(_config.CACHE_MAX_ENTRIES),
            "ttl_sec": int(_config.CACHE_TTL_SEC),
            "fresh_sec": int(_config.CACHE_FRESH_SEC),
            "file": _db_file(),
            "hits_fresh": _hits_fresh,
            "hits_revalidated": _hits_revalidated,
            "misses": _misses,
            "link_hits": _link_hits,
            "link_misses": _link_misses,
        }


def _reset_for_tests() -> None:
    """Test-only: drop in-memory state + counters (db files untouched)."""
    global _store, _hits_fresh, _hits_revalidated, _misses, _link_hits, _link_misses
    with _lock:
        _close_locked()
        _store = None
        _hits_fresh = 0
        _hits_revalidated = 0
        _misses = 0
        _link_hits = 0
        _link_misses = 0
