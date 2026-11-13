"""Reranking the top 50 of a first stage with a cross-encoder (chapter 29).

Usage: uv run --group neural python -m guide_nlp.chapter29.rerank
"""

import numpy as np
from sentence_transformers import CrossEncoder

from guide_nlp.chapter13.metrics import mean_metrics, rank_metrics
from guide_nlp.chapter28.vectors import SOLON, corpus_vectors
from guide_nlp.corpus.load import load_articles, load_queries


def main() -> None:
    articles, queries = load_articles(), load_queries()
    ids = [a.pageid for a in articles]
    documents = [" ".join(a.text.split()[:300]) for a in articles]
    doc_vectors, query_vectors = corpus_vectors(SOLON)
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)
    before, after = [], []
    for q, query_vector in zip(queries, query_vectors, strict=True):
        first_stage_scores = doc_vectors @ query_vector
        top = np.argsort(-first_stage_scores)[:50]
        scores = reranker.predict([(q.query, documents[i]) for i in top], batch_size=64)
        reranked = top[np.argsort(-scores)]
        before.append(rank_metrics([ids[i] for i in top], set(q.relevant)))
        after.append(rank_metrics([ids[i] for i in reranked], set(q.relevant)))
    print(f"dense top 50:          nDCG@10={mean_metrics(before)['ndcg']:.3f}")
    print(f"+ bge-reranker-v2-m3:  nDCG@10={mean_metrics(after)['ndcg']:.3f}")


if __name__ == "__main__":
    main()
