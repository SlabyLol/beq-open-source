"""Beq Python client – call your own Beq API from any Python app."""

from .client import Beq, BeqError, BeqAuthError, BeqRateLimitError
from .languages import LANGUAGES, resolve_language

__all__ = [
    "Beq",
    "BeqError",
    "BeqAuthError",
    "BeqRateLimitError",
    "LANGUAGES",
    "resolve_language",
]
__version__ = "0.1.0"
