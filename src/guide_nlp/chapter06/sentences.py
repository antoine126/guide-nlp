"""Sentence segmentation: naive split, rules, pySBD and spaCy (chapter 6).

Usage: uv run python -m guide_nlp.chapter06.sentences
"""

import re

import pysbd
import spacy

TEXT = (
    "M. Dupont est arrivé à 9 h. Il a payé 3,14 €, etc. "
    "Puis il est reparti vers St. Malo. Était-ce prévu ?"
)

ABBREVIATIONS = {"M", "Mme", "Mlle", "Dr", "Pr", "St", "Ste", "cf", "ex"}
SENT_END = re.compile(r"[.!?…]+\s+")


def split_sentences(text: str) -> list[str]:
    """Rule-based sentence splitter with an abbreviation list."""
    sentences, start = [], 0
    for m in SENT_END.finditer(text):
        next_char = text[m.end() : m.end() + 1]
        words = text[start : m.start() + 1].split()
        last_word = words[-1].rstrip(".!?…") if words else ""
        if not next_char.isupper() or last_word in ABBREVIATIONS:
            continue
        sentences.append(text[start : m.end()].strip())
        start = m.end()
    sentences.append(text[start:].strip())
    return [s for s in sentences if s]


def main() -> None:
    print(TEXT.split(". "))
    # ['M', 'Dupont est arrivé à 9 h', 'Il a payé 3,14 €, etc',
    #  'Puis il est reparti vers St', 'Malo', 'Était-ce prévu ?']

    print(split_sentences(TEXT))
    # ['M. Dupont est arrivé à 9 h.', 'Il a payé 3,14 €, etc.',
    #  'Puis il est reparti vers St. Malo.', 'Était-ce prévu ?']

    segmenter = pysbd.Segmenter(language="fr", clean=False)
    print(segmenter.segment(TEXT))
    # ['M. Dupont est arrivé à 9 h. ', 'Il a payé 3,14 €, etc. ',
    #  'Puis il est reparti vers St. ', 'Malo. ', 'Était-ce prévu ?']

    nlp = spacy.load("fr_core_news_sm")
    print([s.text for s in nlp(TEXT).sents])
    # ['M. Dupont est arrivé à 9', 'h.', 'Il a payé 3,14 €, etc.',
    #  'Puis il est reparti vers St. Malo.', 'Était-ce prévu ?']


if __name__ == "__main__":
    main()
