"""Remove markup and mask URLs, e-mails and numbers (chapter 6).

Usage: uv run python -m guide_nlp.chapter06.clean
"""

import re

from guide_nlp.text.normalize import normalize_text

HTML_TAG = re.compile(r"<[^>]+>")
URL = re.compile(r"https?://\S+|www\.\S+")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
NUMBER = re.compile(r"\b\d+(?:[.,]\d+)?\b")


def clean(text: str, *, mask_numbers: bool = False) -> str:
    """Remove markup and replace URLs, e-mails (and numbers) by placeholders."""
    text = HTML_TAG.sub(" ", text)
    text = URL.sub(" <URL> ", text)
    text = EMAIL.sub(" <EMAIL> ", text)
    if mask_numbers:
        text = NUMBER.sub(" <NUM> ", text)
    return normalize_text(text)


if __name__ == "__main__":
    print(
        clean(
            "<p>Voir https://exemple.fr/doc?id=3 ou écrire à info@exemple.fr avant le 12 mai.</p>",
            mask_numbers=True,
        )
    )
    # Voir <URL> ou écrire à <EMAIL> avant le <NUM> mai.
