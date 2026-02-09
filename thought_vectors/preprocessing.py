from __future__ import annotations

import re
import unicodedata

_APOSTROPHE_CHARS = {
    "’",
    "‘",
    "`",
    "´",
    "ʼ",
    "ʹ",
    "＇",
}

_DASH_CHARS = {
    "–",
    "—",
    "−",
}

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_apostrophes(text: str) -> str:
    """Normalize mixed apostrophe-like characters to plain ASCII apostrophe."""
    out = text
    for char in _APOSTROPHE_CHARS:
        out = out.replace(char, "'")
    return out


def normalize_text(text: str) -> str:
    """Lightweight text cleanup for training consistency.

    - Unicode normalize (NFKC)
    - Normalize apostrophes and dash variants
    - Drop control characters
    - Collapse repeated whitespace
    - Trim leading/trailing whitespace
    """
    out = unicodedata.normalize("NFKC", text)
    out = normalize_apostrophes(out)

    for char in _DASH_CHARS:
        out = out.replace(char, "-")

    out = "".join(ch for ch in out if (ch == "\n" or not unicodedata.category(ch).startswith("C")))
    out = _WHITESPACE_RE.sub(" ", out).strip()
    return out
