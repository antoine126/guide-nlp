"""Extractive summarisation with TextRank (chapter 26).

Usage: uv run --group classic python -m guide_nlp.chapter26.textrank
"""

import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from guide_nlp.corpus.load import load_articles

SENT = re.compile(r"(?<=[.!?])\s+")


def textrank(document: str, k: int = 3, damping: float = 0.85) -> str:
    pieces = SENT.split(document.replace("\n", " "))
    sentences = [s.strip() for s in pieces if len(s.split()) >= 4]
    if len(sentences) <= k:
        return " ".join(sentences)
    tfidf = TfidfVectorizer(token_pattern=r"[^\W\d_]+").fit_transform(sentences)
    sim = (tfidf @ tfidf.T).toarray()
    np.fill_diagonal(sim, 0)
    transition = sim / np.maximum(sim.sum(axis=1, keepdims=True), 1e-9)
    rank = np.full(len(sentences), 1 / len(sentences))
    for _ in range(50):
        rank = (1 - damping) / len(sentences) + damping * transition.T @ rank
    return " ".join(sentences[i] for i in sorted(np.argsort(-rank)[:k].tolist()))


if __name__ == "__main__":
    article = next(a for a in load_articles("test") if len(a.text.split()) > 400)
    print(article.title)
    print(textrank(article.text))
