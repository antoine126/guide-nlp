"""Checking a generated watch summary: numbers and names must come from the cited source
(chapter 47).

Usage: uv run python -m guide_nlp.chapter47.watch
"""

import re


def check_sentence(sentence: str, articles: list[str]) -> bool:
    """Every number and every proper name must occur in the cited article."""
    numbers = [int(n) for n in re.findall(r"\[(\d)\]", sentence)]
    cited = [n for n in numbers if 1 <= n <= len(articles)]
    if not cited:
        return False
    source = " ".join(articles[n - 1] for n in cited)
    body = re.sub(r"^\S+\s", "", sentence)  # the first word is capitalised anyway
    facts = re.findall(r"\b\d{3,4}\b|\b[A-ZÉ][\wéèêàç-]{3,}\b", body)
    return all(fact in source for fact in facts)


if __name__ == "__main__":
    articles = [
        "PAF est un langage de programmation créé en 1957 pour le terminal Flexowriter.",
        "StarLogo est un langage de simulation multi-agents développé au MIT.",
    ]
    for sentence in (
        "Le langage PAF, créé en 1957, utilisait un terminal Flexowriter [1].",
        "StarLogo, développé au MIT en 1994, simule des agents [2].",
        "PAF a été créé en 1957.",
    ):
        print(check_sentence(sentence, articles), "|", sentence)
