"""Canonical text normalisation (chapter 3)."""

import re
import unicodedata as ud
from typing import Literal

Form = Literal["NFC", "NFD", "NFKC", "NFKD"]

# Typographic look-alikes mapped to a single canonical character.
_TYPOGRAPHY = str.maketrans({
    "’": "'", "‘": "'", "ʼ": "'",   # apostrophes
    "“": '"', "”": '"',                   # English quotes
    "«": '"', "»": '"',                   # French guillemets
    "‐": "-", "–": "-", "−": "-",    # hyphen, en dash, minus
    " ": " ", " ": " ", " ": " ",    # no-break and thin spaces
})  # fmt: skip
_INVISIBLE = re.compile("[​‌‍⁠﻿]")
_BLANKS = re.compile(r"[ \t]+")


def normalize_text(text: str, *, form: Form = "NFC", unify_typography: bool = True) -> str:
    """Return a canonical version of ``text`` for downstream processing."""
    text = ud.normalize(form, text)
    if unify_typography:
        text = text.translate(_TYPOGRAPHY)
    text = _INVISIBLE.sub("", text)
    text = "".join(c for c in text if ud.category(c) != "Cc" or c in "\n\t")
    text = _BLANKS.sub(" ", text)
    return text.strip()
