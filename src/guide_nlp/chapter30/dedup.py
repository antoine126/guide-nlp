"""Semantic near-duplicates and novelty detection with embeddings (chapter 30).

Usage: uv run --group neural python -m guide_nlp.chapter30.dedup
"""

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from guide_nlp.chapter28.vectors import SOLON, corpus_vectors
from guide_nlp.corpus.load import RUBRICS, WikiArticle, load_articles


def near_duplicates(vectors: np.ndarray, threshold: float = 0.85) -> np.ndarray:
    similarities = vectors @ vectors.T
    np.fill_diagonal(similarities, 0)
    return np.argwhere(np.triu(similarities, 1) >= threshold)


def novelty_auc(articles: list[WikiArticle], vectors: np.ndarray, new_rubric: str) -> float:
    """Index the training articles of five rubrics; does the mean distance of a test
    article to its 5 nearest neighbours reveal the sixth rubric?"""
    known_vectors = vectors[
        [i for i, a in enumerate(articles) if a.split == "train" and a.rubric != new_rubric]
    ]
    test = [i for i, a in enumerate(articles) if a.split == "test"]
    new_vectors = vectors[test]
    is_new = [articles[i].rubric == new_rubric for i in test]
    index = NearestNeighbors(n_neighbors=5, metric="cosine").fit(known_vectors)
    distances = index.kneighbors(new_vectors)[0].mean(axis=1)
    return float(roc_auc_score(is_new, distances))


def main() -> None:
    articles = load_articles()
    vectors, _ = corpus_vectors(SOLON)
    pairs = near_duplicates(vectors)
    print(f"{len(pairs)} pairs above 0.85")
    for i, j in pairs[:5]:
        print(f"  {articles[i].title} / {articles[j].title}: {vectors[i] @ vectors[j]:.3f}")
    for rubric in RUBRICS:
        print(f"novelty of '{rubric}': AUC = {novelty_auc(articles, vectors, rubric):.3f}")


if __name__ == "__main__":
    main()
