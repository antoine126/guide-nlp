"""Snowball stemming of French words (chapter 6).

Usage: uv run python -m guide_nlp.chapter06.stemming
"""

import snowballstemmer


def main() -> None:
    stemmer = snowballstemmer.stemmer("french")
    words = ["mangeaient", "mange", "manger", "mangeons", "mangeur", "université", "univers"]
    print({w: stemmer.stemWord(w) for w in words})
    # {'mangeaient': 'mang', 'mange': 'mang', 'manger': 'mang',
    #  'mangeons': 'mangeon', 'mangeur': 'mangeur',
    #  'université': 'univers', 'univers': 'univer'}


if __name__ == "__main__":
    main()
