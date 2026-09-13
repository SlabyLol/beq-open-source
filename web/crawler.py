"""
Beq Crawl Bot — fetch public web pages and store text for search / training.

Safety limits (free-tier friendly):
  - only http/https
  - max body size, timeout
  - no login walls / no private data goals
  - stores plain text under data/crawl_store/
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[1]
STORE_DIR = REPO_ROOT / "data" / "crawl_store"
INDEX_PATH = STORE_DIR / "index.jsonl"
CORPUS_PATH = REPO_ROOT / "data" / "crawl_corpus.txt"

MAX_BYTES = 80_000
TIMEOUT = 10.0
MAX_DOCS = 200
USER_AGENT = "BeqCrawlBot/1.0 (+https://beq.onrender.com; open-source research bot)"


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript", "svg", "nav", "footer", "header"):
            self._skip += 1

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
        raw = " ".join(self._chunks)
        raw = re.sub(r"\s+", " ", raw)
        return raw.strip()


def _ok_url(url: str) -> bool:
    try:
        p = urlparse(url.strip())
    except Exception:
        return False
    if p.scheme not in ("http", "https"):
        return False
    if not p.netloc:
        return False
    host = p.netloc.lower()
    if host in ("localhost", "127.0.0.1") or host.startswith("192.168.") or host.startswith("10."):
        return False
    return True


def _html_to_text(html: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(html)
        parser.close()
    except Exception:
        pass
    return parser.text()[:MAX_BYTES]


def _ensure_store() -> None:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    if not INDEX_PATH.exists():
        INDEX_PATH.write_text("", encoding="utf-8")


def list_docs(limit: int = 50) -> list[dict]:
    _ensure_store()
    rows: list[dict] = []
    if not INDEX_PATH.exists():
        return rows
    for line in INDEX_PATH.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows[-limit:]


def _append_index(entry: dict) -> None:
    _ensure_store()
    with INDEX_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    # trim if too many
    rows = list_docs(limit=10_000)
    if len(rows) > MAX_DOCS:
        keep = rows[-MAX_DOCS:]
        INDEX_PATH.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in keep) + "\n",
            encoding="utf-8",
        )


def crawl_url(url: str, save_corpus: bool = True) -> dict:
    """Fetch a public URL and store extracted text. Returns status dict."""
    url = (url or "").strip()
    if not _ok_url(url):
        return {"ok": False, "error": "Invalid or blocked URL (http/https public only)."}
    if httpx is None:
        return {"ok": False, "error": "httpx not installed. Add httpx to requirements."}

    try:
        with httpx.Client(timeout=TIMEOUT, follow_redirects=True, headers={"User-Agent": USER_AGENT}) as client:
            resp = client.get(url)
            resp.raise_for_status()
            ctype = (resp.headers.get("content-type") or "").lower()
            raw = resp.content[: MAX_BYTES + 5000]
            text: str
            if "html" in ctype or url.rstrip("/").endswith((".html", ".htm")) or b"<html" in raw[:500].lower():
                text = _html_to_text(raw.decode(resp.encoding or "utf-8", errors="ignore"))
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
        "title": text[:80],
        "text": text[:MAX_BYTES],
        "chars": len(text),
        "crawled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _append_index(entry)

    if save_corpus:
        CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with CORPUS_PATH.open("a", encoding="utf-8") as f:
            f.write(f"\n\n# source: {url}\n{text[:8000]}\n")

    return {
        "ok": True,
        "id": doc_id,
        "url": url,
        "chars": entry["chars"],
        "preview": text[:240],
    }


def search_store(query: str, limit: int = 5) -> list[dict]:
    """Keyword search over stored crawl docs."""
    q = (query or "").strip().lower()
    if not q:
        return []
    tokens = [t for t in re.split(r"\W+", q) if len(t) > 2]
    if not tokens:
        tokens = [q]
    scored: list[tuple[float, dict]] = []
    for doc in list_docs(limit=MAX_DOCS):
        blob = ((doc.get("title") or "") + " " + (doc.get("text") or "")).lower()
        hits = sum(1 for t in tokens if t in blob)
        if hits <= 0:
            continue
        score = hits / len(tokens)
        # prefer denser early mention
        pos = min((blob.find(t) for t in tokens if t in blob), default=0)
        score += max(0.0, 0.2 - pos / 5000.0)
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    for score, doc in scored[:limit]:
        text = doc.get("text") or ""
        out.append({
            "score": round(score, 3),
            "url": doc.get("url"),
            "id": doc.get("id"),
            "snippet": text[:400],
            "chars": doc.get("chars"),
        })
    return out
