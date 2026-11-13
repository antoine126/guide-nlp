"""Okapi BM25 from scratch over an inverted index, then with bm25s (chapter 13).

Usage: uv run --group classic python -m guide_nlp.chapter13.bm25
"""

import heapq
import math
from collections import Counter, defaultdict

import bm25s
import numpy as np

from guide_nlp.chapter13.metrics import mean_metrics, rank_metrics
from guide_nlp.corpus.load import load_articles, load_queries, words


class BM25Index:
    """Okapi BM25 over an inverted index (Lucene's IDF variant)."""

    def __init__(self, tokenized: list[list[str]], k1: float = 1.2, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths = [len(doc) for doc in tokenized]
        self.avg_len = sum(self.lengths) / len(tokenized)
        for doc_id, doc in enumerate(tokenized):
            for term, tf in Counter(doc).items():
                self.postings[term].append((doc_id, tf))
        n = len(tokenized)
        self.idf = {
            t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in self.postings.items()
        }

    def search(self, query: list[str], top_k: int = 10) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        for term in query:
            for doc_id, tf in self.postings.get(term, ()):
                relative = self.lengths[doc_id] / self.avg_len
                norm = self.k1 * (1 - self.b + self.b * relative)
                scores[doc_id] += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
        return heapq.nlargest(top_k, scores.items(), key=lambda kv: kv[1])


def main() -> None:
    articles, queries = load_articles(), load_queries()
    ids = [a.pageid for a in articles]
    tokenized_docs = [words(a.text) for a in articles]
    tokenized_queries = [words(q.query) for q in queries]

    index = BM25Index(tokenized_docs)
    rows = [
        rank_metrics([ids[d] for d, _ in index.search(q, 100)], set(query.relevant))
        for q, query in zip(tokenized_queries, queries, strict=True)
    ]
    m = mean_metrics(rows)
    print(
        f"BM25 (k1=1.2, b=0.75): MRR={m['mrr']:.3f} R@10={m['recall']:.3f} nDCG@10={m['ndcg']:.3f}"
    )
    # Book, table 13.1 (lowercase only): MRR=0.838 R@10=0.938 nDCG@10=0.860

    retriever = bm25s.BM25(method="lucene", k1=1.2, b=0.75)
    retriever.index(tokenized_docs, show_progress=False)
    doc_ids, _ = retriever.retrieve(tokenized_queries, k=10, show_progress=False)
    same = np.mean(
        [
            set(r.tolist()) == {d for d, _ in index.search(q, 10)}
            for r, q in zip(doc_ids, tokenized_queries, strict=True)
        ]
    )
    print(f"bm25s returns the same top-10 for {same:.0%} of the queries")


if __name__ == "__main__":
    main()
