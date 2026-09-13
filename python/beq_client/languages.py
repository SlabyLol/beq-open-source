"""Language codes for prompt prefixes."""

from __future__ import annotations

LANGUAGES: dict[str, dict[str, str]] = {
    "en": {"name": "English", "prefix": "Answer in English.\n"},
    "de": {"name": "Deutsch", "prefix": "Antworte auf Deutsch.\n"},
    "fr": {"name": "Français", "prefix": "Réponds en français.\n"},
    "es": {"name": "Español", "prefix": "Responde en español.\n"},
    "it": {"name": "Italiano", "prefix": "Rispondi in italiano.\n"},
    "pt": {"name": "Português", "prefix": "Responda em português.\n"},
    "nl": {"name": "Nederlands", "prefix": "Antwoord in het Nederlands.\n"},
    "pl": {"name": "Polski", "prefix": "Odpowiedz po polsku.\n"},
    "ru": {"name": "Русский", "prefix": "Ответь на русском.\n"},
    "ja": {"name": "日本語", "prefix": "日本語で答えてください。\n"},
    "zh": {"name": "中文", "prefix": "请用中文回答。\n"},
    "ko": {"name": "한국어", "prefix": "한국어로 답하세요.\n"},
    "tr": {"name": "Türkçe", "prefix": "Türkçe cevap ver.\n"},
    "ar": {"name": "العربية", "prefix": "أجب بالعربية.\n"},
}


def resolve_language(code: str | None) -> tuple[str, str]:
    if not code:
        return "en", ""
    c = code.strip().lower().replace("_", "-")
    if "-" in c:
        c = c.split("-", 1)[0]
    if c in LANGUAGES:
        return c, LANGUAGES[c]["prefix"]
    return c, f"Answer in {code}.\n"


def list_languages() -> list[dict[str, str]]:
    return [{"code": k, "name": v["name"]} for k, v in LANGUAGES.items()]
