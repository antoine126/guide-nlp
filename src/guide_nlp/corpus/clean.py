"""Boilerplate removal for Wikipedia extracts (chapter 10)."""

import re

TAIL_SECTION = re.compile(
    r"^(Notes et références|Notes|Références|Voir aussi|Annexes|Bibliographie|"
    r"Liens externes|Articles connexes|Sources)\s*$",
    re.MULTILINE,
)


def strip_tail(text: str) -> str:
    """Cut the text at the first reference-like section heading."""
    match = TAIL_SECTION.search(text)
    return text[: match.start()].rstrip() if match else text
