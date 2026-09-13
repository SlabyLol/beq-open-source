"""HTTP client for Beq API."""

from __future__ import annotations

from typing import Any

import requests

from .languages import list_languages, resolve_language


class BeqError(Exception):
    def __init__(self, message: str, status: int | None = None, body: Any = None):
        super().__init__(message)
        self.status = status
        self.body = body


class BeqAuthError(BeqError):
    pass


class BeqRateLimitError(BeqError):
    pass


class Beq:
    """
    Python plugin for the Beq API.

        from beq_client import Beq
        beq = Beq(base_url="https://beq.onrender.com", api_key="beq_...", language="de")
        print(beq.generate("Hallo"))
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        *,
        language: str = "en",
        timeout: float = 300.0,
        default_max_tokens: int = 80,
        default_temperature: float = 0.8,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = (api_key or "").strip() or None
        self.language = language
        self.timeout = timeout
        self.default_max_tokens = default_max_tokens
        self.default_temperature = default_temperature
        self._session = requests.Session()

    def _request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        """Send a request and turn network timeouts into a useful SDK error."""
        try:
            return self._session.request(method, url, timeout=self.timeout, **kwargs)
        except requests.Timeout as exc:
            raise BeqError(
                f"Request timed out after {self.timeout} seconds. "
                "The Beq server may be waking up; retry or increase the timeout."
            ) from exc

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    def _raise(self, r: requests.Response) -> None:
        try:
            body = r.json()
        except Exception:
            body = r.text
        msg = body.get("error") if isinstance(body, dict) else str(body)
        if r.status_code in (401, 403):
            raise BeqAuthError(msg or "Auth failed", r.status_code, body)
        if r.status_code == 429:
            raise BeqRateLimitError(msg or "Rate / token limit", r.status_code, body)
        if r.status_code >= 400:
            raise BeqError(msg or f"HTTP {r.status_code}", r.status_code, body)

    def status(self) -> dict[str, Any]:
        r = self._request("GET", f"{self.base_url}/api/status")
        self._raise(r)
        return r.json()

    def generate(
        self,
        prompt: str,
        *,
        language: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        use_language_prefix: bool = True,
    ) -> str:
        data = self.generate_full(
            prompt,
            language=language,
            max_tokens=max_tokens,
            temperature=temperature,
            use_language_prefix=use_language_prefix,
        )
        return data.get("new_text") or data.get("generated") or ""

    def generate_full(
        self,
        prompt: str,
        *,
        language: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        use_language_prefix: bool = True,
    ) -> dict[str, Any]:
        lang = language if language is not None else self.language
        code, prefix = resolve_language(lang)
        text = prefix + prompt if (use_language_prefix and prefix) else prompt
        payload = {
            "prompt": text,
            "max_tokens": max_tokens if max_tokens is not None else self.default_max_tokens,
            "temperature": temperature if temperature is not None else self.default_temperature,
        }
        r = self._request(
            "POST",
            f"{self.base_url}/api/generate",
            json=payload,
            headers=self._headers(),
        )
        self._raise(r)
        out = r.json()
        out["_language"] = code
        return out

    def chat(
        self,
        message: str,
        *,
        language: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        return self.generate(
            message,
            language=language,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def set_language(self, language: str) -> None:
        self.language = language

    @staticmethod
    def languages() -> list[dict[str, str]]:
        return list_languages()
