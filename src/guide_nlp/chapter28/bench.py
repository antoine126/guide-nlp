"""Benchmark embedding models on our 112 queries, against BM25 (chapter 28).

Usage: uv run --group neural python -m guide_nlp.chapter28.bench
"""

import bm25s
import numpy as np

from guide_nlp.chapter13.metrics import (
    evaluate_scores,
    mean_metrics,
    paired_bootstrap,
    rank_metrics,
)
from guide_nlp.chapter28.vectors import EmbeddingModel, corpus_vectors
from guide_nlp.corpus.load import load_articles, load_queries, words

MODELS = [  # name, query prefix, document prefix
    EmbeddingModel("sentence-transformers/paraphrase-multilingual-mpnet-base-v2"),
    EmbeddingModel("intfloat/multilingual-e5-large", "query: ", "passage: "),
    EmbeddingModel("BAAI/bge-m3"),
    EmbeddingModel("Snowflake/snowflake-arctic-embed-l-v2.0", "query: "),
    EmbeddingModel("OrdalieTech/Solon-embeddings-large-0.1", "query : "),
]


def bm25_rows() -> list[dict[str, float]]:
    articles, queries = load_articles(), load_queries()
    ids = [a.pageid for a in articles]
    retriever = bm25s.BM25(method="lucene", k1=1.2, b=0.75)
    retriever.index([words(a.text) for a in articles], show_progress=False)
    results, _ = retriever.retrieve([words(q.query) for q in queries], k=100, show_progress=False)
    return [
        rank_metrics([ids[i] for i in r], set(q.relevant))
        for r, q in zip(results.tolist(), queries, strict=True)
    ]


def main() -> None:
    ids = [a.pageid for a in load_articles()]
    relevant = [set(q.relevant) for q in load_queries()]
    baseline = bm25_rows()
    print(f"{'BM25':<60} nDCG@10={mean_metrics(baseline)['ndcg']:.3f}")
    for model in MODELS:
        docs, queries = corpus_vectors(model)
        rows = evaluate_scores(queries @ docs.T, ids, relevant)
        diff, low, high = paired_bootstrap([r["ndcg"] for r in rows], [r["ndcg"] for r in baseline])
        print(
            f"{model.name:<60} nDCG@10={mean_metrics(rows)['ndcg']:.3f} "
            f"vs BM25 {diff:+.3f} [{low:+.3f}, {high:+.3f}]"
        )
    # Book, table 28.2: BM25 0.860, mpnet 0.727, e5-large 0.964, bge-m3 0.957, Solon 0.970...


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    main()
