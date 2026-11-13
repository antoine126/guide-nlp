"""Heuristic quality filters (chapter 10)."""

import re

WORD = re.compile(r"[^\W\d_]+(?:['-][^\W\d_]+)*")
FUNCTION_WORDS = set(
    "le la les de des du un une et en à au aux que qui dans pour sur par "
    "avec est sont il elle ils elles ce cette ces se sa son ses ne pas "
    "plus ou mais comme d l".split()
)


def quality_issue(text: str) -> str | None:
    """Return the reason why a text should be rejected, if any."""
    words = [w.lower() for w in WORD.findall(text)]
    if len(words) < 50:
        return "trop court"
    letters_or_spaces = sum(c.isalpha() or c.isspace() for c in text)
    if 1 - letters_or_spaces / len(text) > 0.25:
        return "trop de symboles"
    lines = [line for line in text.split("\n") if line.strip()]
    if sum(len(line.split()) < 5 for line in lines) / len(lines) > 0.70:
        return "liste, pas de la prose"
    tokens = [p for w in words for p in re.split(r"['-]", w)]
    if sum(t in FUNCTION_WORDS for t in tokens) / len(tokens) < 0.20:
        return "peu de mots grammaticaux"
    return None
