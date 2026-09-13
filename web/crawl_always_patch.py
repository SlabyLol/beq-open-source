"""Patch: always_search for Beq crawler. Auto-applied from searcher import."""
from web import crawler, settings_store

SETTING = "crawl_always_search"


def always_search_enabled() -> bool:
    try:
        return (settings_store.get_setting(SETTING, "1") or "1") not in ("0", "false", "off", "no")
    except Exception:
        return True


def set_always_search(on: bool) -> None:
    settings_store.set_setting(SETTING, "1" if on else "0")


crawler.always_search_enabled = always_search_enabled
crawler.set_always_search = set_always_search

_orig_status = crawler.status


def status():
    s = _orig_status()
    s["always_search"] = always_search_enabled()
    return s


crawler.status = status
