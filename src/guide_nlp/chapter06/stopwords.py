"""spaCy's French stop-word list, and the negations it contains (chapter 6).

Usage: uv run python -m guide_nlp.chapter06.stopwords
"""

from spacy.lang.fr.stop_words import STOP_WORDS


def main() -> None:
    print(len(STOP_WORDS))  # 507 (spaCy 3.8)
    critical = {"ne", "pas", "plus", "sans", "non", "jamais", "rien"}
    print(sorted(critical & STOP_WORDS))  # ['ne', 'pas', 'plus', 'sans']


if __name__ == "__main__":
    main()
