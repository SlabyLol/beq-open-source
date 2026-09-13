"""
Beq Crawl Bot — real web crawler + search crawl for Beq knowledge.

Features:
  - Crawl any public http/https URL
  - Web-search crawl (DuckDuckGo HTML results; Google-style queries)
  - Random crawl from seed list
  - On/Off for using crawl/search in chat answers
  - Persistent storage on Render via DATABASE_URL (Neon/Postgres) or local SQLite

Admin controls these from /admin.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse, urljoin, quote_plus

from web import settings_store

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[1]
STORE_DIR = REPO_ROOT / "data" / "crawl_store"
INDEX_PATH = STORE_DIR / "index.jsonl"
CORPUS_PATH = REPO_ROOT / "data" / "crawl_corpus.txt"

MAX_BYTES = 80_000
TIMEOUT = 12.0
MAX_DOCS = 500
USER_AGENT = (
    "Mozilla/5.0 (compatible; BeqCrawlBot/1.1; +https://beq.onrender.com; "
    "open-source knowledge bot)"
)

SETTING_ENABLED = "crawl_enabled"
SETTING_WEB_IN_CHAT = "crawl_web_in_chat"
SETTING_AUTO_SEARCH = "crawl_auto_search"

SEED_URLS = [
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://en.wikipedia.org/wiki/Python_(programming_language)",
    "https://en.wikipedia.org/wiki/Solar_System",
    "https://en.wikipedia.org/wiki/Earth",
    "https://en.wikipedia.org/wiki/Mathematics",
    "https://en.wikipedia.org/wiki/Computer",
    "https://en.wikipedia.org/wiki/Internet",
    "https://en.wikipedia.org/wiki/Physics",
    "https://en.wikipedia.org/wiki/Biology",
    "https://en.wikipedia.org/wiki/History",
    "https://simple.wikipedia.org/wiki/Main_Page",
]


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip = 0
        self._links: list[str] = []
        self._base = ""

    def set_base(self, base: str) -> None:
        self._base = base

    def handle_starttag(self, tag, attrs):
        ad = dict(attrs)
        if tag in ("script", "style", "noscript", "svg", "nav", "footer", "header"):
            self._skip += 1
        if tag == "a" and "href" in ad:
            href = ad["href"]
            if href and not href.startswith(("#", "javascript:", "mailto:")):
                full = urljoin(self._base, href)
                if full.startswith("http"):
                    self._links.append(full)

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript", "svg", "nav", "footer", "header"):
            self._skip = max(0, self._skip - 1)

    def handle_data(self, data):
        if self._skip:
            return
        t = data.strip()
        if t:
            self._chunks.append(t)

    def text(self) -> str:
        return re.sub(r"\s+", " ", " ".join(self._chunks)).strip()


def is_enabled() -> bool:
    try:
        return (settings_store.get_setting(SETTING_ENABLED, "1") or "1") not in ("0", "false", "off", "no")
    except Exception:
        return True


def set_enabled(on: bool) -> None:
    settings_store.set_setting(SETTING_ENABLED, "1" if on else "0")


def web_in_chat_enabled() -> bool:
    try:
        return (settings_store.get_setting(SETTING_WEB_IN_CHAT, "1") or "1") not in ("0", "false", "off", "no")
    except Exception:
        return True


def set_web_in_chat(on: bool) -> None:
    settings_store.set_setting(SETTING_WEB_IN_CHAT, "1" if on else "0")


def auto_search_enabled() -> bool:
    try:
        return (settings_store.get_setting(SETTING_AUTO_SEARCH, "1") or "1") not in ("0", "false", "off", "no")
    except Exception:
        return True


def set_auto_search(on: bool) -> None:
    settings_store.set_setting(SETTING_AUTO_SEARCH, "1" if on else "0")


def _ok_url(url: str) -> bool:
    try:
        p = urlparse(url.strip())
    except Exception:
        return False
    if p.scheme not in ("http", "https") or not p.netloc:
        return False
    host = p.netloc.lower()
    if host in ("localhost", "127.0.0.1") or host.startswith(("192.168.", "10.", "0.")):
        return False
    return True


def _html_to_text(html: str, base: str = "") -> tuple[str, list[str]]:
    parser = _TextExtractor()
    if base:
        parser.set_base(base)
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        pass
    return parser.text()[:MAX_BYTES], parser._links[:50]


def _ensure_store() -> None:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    if not INDEX_PATH.exists():
        INDEX_PATH.write_text("", encoding="utf-8")


def _init_crawl_table() -> None:
    try:
        settings_store.init_settings_table()
    except Exception:
        pass
    if settings_store._is_postgres():
        conn = settings_store._pg_conn()
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS crawl_docs (
                id TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                title TEXT,
                text TEXT NOT NULL,
                chars INTEGER,
                crawled_at TEXT
            )
            """
        )
        conn.commit()
        cur.close()
        conn.close()
    else:
        with settings_store._sqlite_conn() as c:
            c.execute(
                """
                CREATE TABLE IF NOT EXISTS crawl_docs (
                    id TEXT PRIMARY KEY,
                    url TEXT NOT NULL,
                    title TEXT,
                    text TEXT NOT NULL,
                    chars INTEGER,
                    crawled_at TEXT
                )
                """
            )
            c.commit()


def _db_save(entry: dict) -> None:
    _init_crawl_table()
    if settings_store._is_postgres():
        conn = settings_store._pg_conn()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO crawl_docs (id, url, title, text, chars, crawled_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
              title = EXCLUDED.title,
              text = EXCLUDED.text,
              chars = EXCLUDED.chars,
              crawled_at = EXCLUDED.crawled_at
            """,
            (
                entry["id"],
                entry["url"],
                entry.get("title") or "",
                entry["text"],
                entry.get("chars") or len(entry["text"]),
                entry.get("crawled_at") or "",
            ),
        )
        conn.commit()
        cur.close()
        conn.close()
        return
    with settings_store._sqlite_conn() as c:
        c.execute(
            """
            INSERT INTO crawl_docs (id, url, title, text, chars, crawled_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              title = excluded.title,
              text = excluded.text,
              chars = excluded.chars,
              crawled_at = excluded.crawled_at
            """,
            (
                entry["id"],
                entry["url"],
                entry.get("title") or "",
                entry["text"],
                entry.get("chars") or len(entry["text"]),
                entry.get("crawled_at") or "",
            ),
        )
        c.commit()


def _db_list(limit: int = 50) -> list[dict]:
    _init_crawl_table()
    rows: list[dict] = []
    try:
        if settings_store._is_postgres():
            conn = settings_store._pg_conn()
            cur = conn.cursor()
            cur.execute(
                "SELECT id, url, title, text, chars, crawled_at FROM crawl_docs ORDER BY crawled_at DESC LIMIT %s",
                (limit,),
            )
            for r in cur.fetchall():
                rows.append(
                    {
                        "id": r[0],
                        "url": r[1],
                        "title": r[2],
                        "text": r[3],
                        "chars": r[4],
                        "crawled_at": r[5],
                    }
                )
            cur.close()
            conn.close()
        else:
            with settings_store._sqlite_conn() as c:
                for r in c.execute(
                    "SELECT id, url, title, text, chars, crawled_at FROM crawl_docs ORDER BY crawled_at DESC LIMIT ?",
                    (limit,),
                ):
                    rows.append(dict(r))
    except Exception as e:
        print(f"[crawler] db list: {e}")
    return rows


def _db_search(query: str, limit: int = 5) -> list[dict]:
    q = (query or "").strip().lower()
    if not q:
        return []
    tokens = [t for t in re.split(r"\W+", q) if len(t) > 2] or [q]
    scored: list[tuple[float, dict]] = []
    for doc in _db_list(limit=MAX_DOCS):
        blob = ((doc.get("title") or "") + " " + (doc.get("text") or "")).lower()
        hits = sum(1 for t in tokens if t in blob)
        if hits <= 0:
            continue
        score = hits / len(tokens)
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    for score, doc in scored[:limit]:
        out.append(
            {
                "score": round(score, 3),
                "url": doc.get("url"),
                "id": doc.get("id"),
                "snippet": (doc.get("text") or "")[:400],
                "chars": doc.get("chars"),
            }
        )
    return out


def list_docs(limit: int = 50) -> list[dict]:
    rows = _db_list(limit=limit)
    if rows:
        return rows
    # file fallback
    _ensure_store()
    out: list[dict] = []
    if INDEX_PATH.exists():
        for line in INDEX_PATH.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out[-limit:]


def _append_file_index(entry: dict) -> None:
    _ensure_store()
    with INDEX_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def crawl_url(url: str, save_corpus: bool = True) -> dict:
    if not is_enabled():
        return {"ok": False, "error": "Crawler is turned OFF in admin."}
    url = (url or "").strip()
    if not _ok_url(url):
        return {"ok": False, "error": "Invalid or blocked URL (public http/https only)."}
    if httpx is None:
        return {"ok": False, "error": "httpx not installed."}

    try:
        with httpx.Client(
            timeout=TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
        ) as client:
            resp = client.get(url)
            resp.raise_for_status()
            ctype = (resp.headers.get("content-type") or "").lower()
            raw = resp.content[: MAX_BYTES + 8000]
            if "html" in ctype or b"<html" in raw[:800].lower():
                text, _links = _html_to_text(
                    raw.decode(resp.encoding or "utf-8", errors="ignore"), base=str(resp.url)
                )
            else:
                text = raw.decode("utf-8", errors="ignore")[:MAX_BYTES]
    except Exception as e:
        return {"ok": False, "error": f"Fetch failed: {e}"}

    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < 40:
        return {"ok": False, "error": "Page had almost no extractable text."}

    doc_id = hashlib.sha1(url.encode()).hexdigest()[:16]
    entry = {
        "id": doc_id,
        "url": url,
        "title": text[:100],
        "text": text[:MAX_BYTES],
        "chars": len(text),
        "crawled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    try:
        _db_save(entry)
    except Exception as e:
        print(f"[crawler] db save failed: {e}")
        _append_file_index(entry)

    if save_corpus:
        try:
            CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with CORPUS_PATH.open("a", encoding="utf-8") as f:
                f.write(f"\n\n# source: {url}\n{text[:6000]}\n")
        except Exception:
            pass

    return {
        "ok": True,
        "id": doc_id,
        "url": url,
        "chars": entry["chars"],
        "preview": text[:240],
        "stored": "database" if settings_store.using_postgres() else "sqlite/file",
    }


def search_store(query: str, limit: int = 5) -> list[dict]:
    hits = _db_search(query, limit=limit)
    if hits:
        return hits
    # file fallback keyword search
    q = (query or "").strip().lower()
    tokens = [t for t in re.split(r"\W+", q) if len(t) > 2] or [q]
    scored: list[tuple[float, dict]] = []
    for doc in list_docs(limit=MAX_DOCS):
        blob = ((doc.get("title") or "") + " " + (doc.get("text") or "")).lower()
        hits_n = sum(1 for t in tokens if t in blob)
        if hits_n <= 0:
            continue
        scored.append((hits_n / len(tokens), doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        {
            "score": round(s, 3),
            "url": d.get("url"),
            "id": d.get("id"),
            "snippet": (d.get("text") or "")[:400],
            "chars": d.get("chars"),
        }
        for s, d in scored[:limit]
    ]


def _extract_ddg_links(html: str) -> list[str]:
    """Parse DuckDuckGo HTML result page for result URLs."""
    links: list[str] = []
    # uddg= redirected urls
    for m in re.finditer(r'uddg=([^&"\']+)', html):
        try:
            u = unquote(m.group(1))
            if u.startswith("http") and "duckduckgo.com" not in u:
                links.append(u)
        except Exception:
            continue
    # plain hrefs
    for m in re.finditer(r'href="(https?://[^"]+)"', html):
        u = m.group(1)
        if "duckduckgo.com" in u or "google.com/search" in u:
            continue
        links.append(u)
    # dedupe keep order
    seen = set()
    out = []
    for u in links:
        if u not in seen and _ok_url(u):
            seen.add(u)
            out.append(u)
    return out[:15]


def web_search_links(query: str, max_links: int = 5) -> list[str]:
    """Search the web (DuckDuckGo HTML) for result links — Google-style query."""
    if httpx is None or not query.strip():
        return []
    q = query.strip()
    urls_try = [
        f"https://html.duckduckgo.com/html/?q={quote_plus(q)}",
        f"https://lite.duckduckgo.com/lite/?q={quote_plus(q)}",
    ]
    links: list[str] = []
    try:
        with httpx.Client(
            timeout=TIMEOUT,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            for search_url in urls_try:
                try:
                    r = client.get(search_url)
                    if r.status_code >= 400:
                        continue
                    links.extend(_extract_ddg_links(r.text))
                    if links:
                        break
                except Exception:
                    continue
    except Exception as e:
        print(f"[crawler] web_search_links: {e}")
    # dedupe
    seen = set()
    out = []
    for u in links:
        if u not in seen:
            seen.add(u)
            out.append(u)
        if len(out) >= max_links:
            break
    return out


def search_and_crawl(query: str, max_pages: int = 3) -> dict:
    """Google-style search query → crawl top result pages into store."""
    if not is_enabled():
        return {"ok": False, "error": "Crawler is OFF", "results": []}
    links = web_search_links(query, max_links=max_pages + 2)
    results = []
    for url in links[:max_pages]:
        r = crawl_url(url)
        results.append({"url": url, **r})
        time.sleep(0.4)
    return {
        "ok": True,
        "query": query,
        "links_found": links,
        "results": results,
        "crawled_ok": sum(1 for x in results if x.get("ok")),
    }


def random_crawl(n: int = 2) -> dict:
    if not is_enabled():
        return {"ok": False, "error": "Crawler is OFF", "results": []}
    n = max(1, min(5, int(n)))
    picks = random.sample(SEED_URLS, k=min(n, len(SEED_URLS)))
    results = []
    for url in picks:
        results.append({"url": url, **crawl_url(url)})
        time.sleep(0.3)
    return {"ok": True, "results": results, "crawled_ok": sum(1 for x in results if x.get("ok"))}


def status() -> dict:
    docs = list_docs(limit=5)
    return {
        "enabled": is_enabled(),
        "web_in_chat": web_in_chat_enabled(),
        "auto_search": auto_search_enabled(),
        "doc_count_sample": len(list_docs(limit=200)),
        "using_postgres": settings_store.using_postgres(),
        "recent": [
            {"url": d.get("url"), "chars": d.get("chars"), "crawled_at": d.get("crawled_at")}
            for d in docs
        ],
    }
