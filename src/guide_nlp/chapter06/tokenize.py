"""Word tokenisation of French: split, ``\\w+``, a dedicated regex, spaCy (chapter 6).

Usage: uv run python -m guide_nlp.chapter06.tokenize
"""

import re

import spacy

SENTENCE = "L'homme d'aujourd'hui a payé 3,14 € à Saint-Étienne, c'est-à-dire trop !"

TOKEN = re.compile(
    r"""
    (?:aujourd'hui|presqu'île|prud'hom\w*)   # lexicalised elisions
  | (?:[^\W\d_]+')                           # elided words: l', d', qu'
  | (?:\d+(?:[.,]\d+)*)                      # numbers: 3,14  1.000
  | (?:[^\W\d_]+(?:-[^\W\d_]+)*)             # words, hyphenated or not
  | (?:[^\w\s])                              # any other symbol
""",
    re.VERBOSE | re.IGNORECASE,
)


def tokenize(text: str) -> list[str]:
    """French word tokens: elisions kept apart, hyphenated words kept whole."""
    return TOKEN.findall(text)


def main() -> None:
    print(SENTENCE.split())
    # ["L'homme", "d'aujourd'hui", 'a', 'payé', '3,14', '€', 'à',
    #  'Saint-Étienne,', "c'est-à-dire", 'trop', '!']

    print(re.findall(r"\w+", SENTENCE))
    # ['L', 'homme', 'd', 'aujourd', 'hui', 'a', 'payé', '3', '14', 'à',
    #  'Saint', 'Étienne', 'c', 'est', 'à', 'dire', 'trop']

    print(tokenize(SENTENCE))
    # ["L'", 'homme', "d'", "aujourd'hui", 'a', 'payé', '3,14', '€', 'à',
    #  'Saint-Étienne', ',', "c'", 'est-à-dire', 'trop', '!']

    nlp = spacy.load("fr_core_news_sm")
    doc = nlp(SENTENCE)
    print([t.text for t in doc])
    # ["L'", 'homme', "d'", "aujourd'hui", 'a', 'payé', '3,14', '€', 'à',
    #  'Saint-Étienne', ',', "c'", 'est', '-', 'à', '-', 'dire', 'trop', '!']


if __name__ == "__main__":
    main()
