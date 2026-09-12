"""
Beq tools: web search + simple page fetch (no paid APIs).
"""

from __future__ import annotations

import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "noscript"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript"):
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            t = data.strip()
            if t:
                self.parts.append(t)


def _fetch(url: str, timeout: int = 8) -> str:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "BeqBot/1.0 (+https://github.com/SlabyLol/beq-private)"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        charset = "utf-8"
        ct = resp.headers.get("Content-Type", "")
        if "charset=" in ct:
            charset = ct.split("charset=")[-1].split(";")[0].strip() or "utf-8"
        return raw.decode(charset, errors="ignore")


def search_web(query: str, max_results: int = 5) -> list[dict]:
    q = urllib.parse.quote_plus(query.strip())
    url = f"https://html.duckduckgo.com/html/?q={q}"
    try:
        html = _fetch(url)
    except Exception as e:
        return [{"title": "Search failed", "url": "", "snippet": str(e)}]

    results = []
    for m in re.finditer(
        r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?class="result__snippet"[^>]*>(.*?)</(?:a|td)',
        html,
        re.I | re.S,
    ):
        href, title, snip = m.group(1), m.group(2), m.group(3)
        title = re.sub(r"<[^>]+>", "", title).strip()
        snip = re.sub(r"<[^>]+>", "", snip).strip()
        if "uddg=" in href:
            href = urllib.parse.unquote(href.split("uddg=")[-1].split("&")[0])
        results.append({"title": title, "url": href, "snippet": snip[:300]})
        if len(results) >= max_results:
            break

    if not results:
        for m in re.finditer(r'href="(https?://[^"]+)"[^>]*>([^<]{10,120})</a>', html):
            href, title = m.group(1), m.group(2).strip()
            if "duckduckgo" in href:
                continue
            results.append({"title": title, "url": href, "snippet": ""})
            if len(results) >= max_results:
                break

    return results or [{"title": "No results", "url": "", "snippet": query}]


def crawl_url(url: str, max_chars: int = 2500) -> str:
    if not url.startswith("http"):
        return "Invalid URL"
    try:
        html = _fetch(url)
        parser = _TextExtractor()
        parser.feed(html)
        text = " ".join(parser.parts)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars]
    except Exception as e:
        return f"Crawl failed: {e}"


def build_context(query: str, use_search: bool = True, crawl: str | None = None) -> str:
    parts = []
    if use_search and query.strip():
        hits = search_web(query, max_results=4)
        lines = ["Search results:"]
        for i, h in enumerate(hits, 1):
            lines.append(f"{i}. {h['title']}: {h['snippet'][:200]}")
        parts.append("\n".join(lines))
    if crawl:
        body = crawl_url(crawl)
        parts.append(f"Page content:\n{body[:1500]}")
    if not parts:
        return ""
    return "\n\n".join(parts) + "\n\n"
