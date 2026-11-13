"""Hybrid search: BM25 + dense, fused by RRF or by a convex combination (chapter 29).

Usage: uv run --group neural python -m guide_nlp.chapter29.hybrid
"""

import bm25s
import numpy as np

from guide_nlp.chapter13.metrics import evaluate_scores, mean_metrics, rank_metrics
from guide_nlp.chapter28.vectors import SOLON, corpus_vectors
from guide_nlp.corpus.load import load_articles, load_queries, words


def rrf(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking, start=1):
            scores[doc] = scores.get(doc, 0.0) + 1 / (k + rank)
    return scores


def minmax(s: np.ndarray) -> np.ndarray:
    low, high = s.min(axis=1, keepdims=True), s.max(axis=1, keepdims=True)
    return (s - low) / np.maximum(high - low, 1e-9)


def bm25_scores(texts: list[str], queries: list[str]) -> np.ndarray:
    """The full (queries x documents) BM25 score matrix."""
    retriever = bm25s.BM25(method="lucene", k1=1.2, b=0.75)
    retriever.index([words(t) for t in texts], show_progress=False)
    docs, scores = retriever.retrieve(
        [words(q) for q in queries], k=len(texts), show_progress=False
    )
    out = np.zeros((len(queries), len(texts)))
    for i in range(len(queries)):
        out[i, docs[i]] = scores[i]
    return out


def main() -> None:
    articles, queries = load_articles(), load_queries()
    ids = [a.pageid for a in articles]
    relevant = [set(q.relevant) for q in queries]
    doc_vectors, query_vectors = corpus_vectors(SOLON)
    dense_scores = query_vectors @ doc_vectors.T
    lexical = bm25_scores([a.text for a in articles], [q.query for q in queries])

    for name, scores in (("BM25", lexical), ("dense (Solon)", dense_scores)):
        print(
            f"{name:<22} nDCG@10={mean_metrics(evaluate_scores(scores, ids, relevant))['ndcg']:.3f}"
        )
    rows = []
    for i, wanted in enumerate(relevant):
        fused = rrf(
            [np.argsort(-dense_scores[i])[:100].tolist(), np.argsort(-lexical[i])[:100].tolist()]
        )
        rows.append(
            rank_metrics(
                [ids[d] for d in sorted(fused, key=fused.__getitem__, reverse=True)], wanted
            )
        )
    print(f"{'RRF':<22} nDCG@10={mean_metrics(rows)['ndcg']:.3f}")
    for alpha in (0.3, 0.5, 0.7):
        convex = alpha * minmax(dense_scores) + (1 - alpha) * minmax(lexical)
        m = mean_metrics(evaluate_scores(convex, ids, relevant))
        print(f"{'convex, alpha=' + str(alpha):<22} nDCG@10={m['ndcg']:.3f}")


if __name__ == "__main__":
    main()
