"""Collocations ranked by pointwise mutual information (chapter 6).

Usage: uv run python -m guide_nlp.chapter06.collocations
"""

import math
from collections import Counter
from itertools import pairwise

from guide_nlp.corpus.load import load_articles, words


def top_collocations(
    sentences: list[list[str]], min_count: int = 5, top: int = 20
) -> list[tuple[tuple[str, str], float]]:
    """Rank adjacent word pairs by pointwise mutual information."""
    unigrams = Counter(w for s in sentences for w in s)
    bigrams = Counter(pair for s in sentences for pair in pairwise(s))
    n_uni, n_bi = sum(unigrams.values()), sum(bigrams.values())
    scores = {}
    for (a, b), n_ab in bigrams.items():
        if n_ab >= min_count:
            p_ab = n_ab / n_bi
            p_a, p_b = unigrams[a] / n_uni, unigrams[b] / n_uni
            scores[(a, b)] = math.log2(p_ab / (p_a * p_b))
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top]


def main() -> None:
    sentences = [words(line) for a in load_articles("train") for line in a.text.split("\n")]
    for (a, b), pmi in top_collocations(sentences, min_count=10):
        print(f"{a + ' ' + b:<30} {pmi:5.2f}")


if __name__ == "__main__":
    main()
