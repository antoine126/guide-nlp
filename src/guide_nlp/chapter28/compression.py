"""Shrinking the vectors: Matryoshka truncation and binary quantisation (chapter 28).

Usage: uv run --group neural python -m guide_nlp.chapter28.compression
"""

import faiss
import numpy as np

from guide_nlp.chapter13.metrics import evaluate_scores, mean_metrics
from guide_nlp.chapter28.vectors import EmbeddingModel, corpus_vectors
from guide_nlp.corpus.load import load_articles, load_queries

MODEL = EmbeddingModel("Snowflake/snowflake-arctic-embed-l-v2.0", "query: ")


def truncate(vectors: np.ndarray, k: int) -> np.ndarray:
    head = vectors[:, :k]
    return head / np.linalg.norm(head, axis=1, keepdims=True)  # renormalise


def binary_search(docs: np.ndarray, queries: np.ndarray, depth: int = 100) -> list[np.ndarray]:
    """Hamming search on sign bits, then rescoring of the candidates in full precision."""
    binary_docs, binary_queries = np.packbits(docs > 0, axis=1), np.packbits(queries > 0, axis=1)
    index = faiss.IndexBinaryFlat(docs.shape[1])
    index.add(binary_docs)
    _, found = index.search(binary_queries, depth)
    candidates = np.asarray(found, dtype=np.int64)
    rankings = []
    for found_row, query in zip(candidates, queries, strict=True):
        row = np.asarray(found_row)
        order = row[np.argsort(-(docs[row] @ query))]  # rescoring with the full vectors
        rankings.append(order)
    return rankings


def main() -> None:
    ids = [a.pageid for a in load_articles()]
    relevant = [set(q.relevant) for q in load_queries()]
    docs, queries = corpus_vectors(MODEL)
    for k in (docs.shape[1], 512, 256, 128):
        rows = evaluate_scores(truncate(queries, k) @ truncate(docs, k).T, ids, relevant)
        print(f"{k:>5} dimensions: nDCG@10={mean_metrics(rows)['ndcg']:.3f}")
    rankings = binary_search(docs, queries)
    scores = np.full((len(queries), len(ids)), -1.0)
    for i, order in enumerate(rankings):
        scores[i, order] = np.linspace(1, 0, len(order))
    rows = evaluate_scores(scores, ids, relevant)
    print(
        f"binary + rescoring: nDCG@10={mean_metrics(rows)['ndcg']:.3f} "
        f"({docs.shape[1] // 8} bytes per document instead of {docs.shape[1] * 4})"
    )


if __name__ == "__main__":
    main()
