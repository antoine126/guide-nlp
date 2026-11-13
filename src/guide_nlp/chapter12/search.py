"""A TF-IDF search engine, evaluated on the 112 queries (chapter 12).

Usage: uv run --group classic python -m guide_nlp.chapter12.search
"""

from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from guide_nlp.chapter13.metrics import mean_metrics, rank_metrics
from guide_nlp.corpus.load import load_articles, load_queries, words


class TfidfSearch:
    def __init__(self, texts: list[str], ids: list[int], **kwargs: Any) -> None:
        self.ids = ids
        self.vectorizer = TfidfVectorizer(**kwargs)
        self.matrix = self.vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        scores = (self.matrix @ self.vectorizer.transform([query]).T).toarray().ravel()
        best = np.argpartition(-scores, top_k)[:top_k]
        best = best[np.argsort(-scores[best])]
        return [(self.ids[i], float(scores[i])) for i in best]


def main() -> None:
    articles = load_articles()
    engine = TfidfSearch(
        [a.text for a in articles],
        [a.pageid for a in articles],
        tokenizer=words,
        token_pattern=None,
        sublinear_tf=True,
    )
    rows = [
        rank_metrics([pid for pid, _ in engine.search(q.query, 100)], set(q.relevant))
        for q in load_queries()
    ]
    m = mean_metrics(rows)
    print(f"TF-IDF (sublinear): MRR={m['mrr']:.3f} R@10={m['recall']:.3f} nDCG@10={m['ndcg']:.3f}")
    # Book, table 12.2: MRR=0.761 R@10=0.911 nDCG@10=0.795


if __name__ == "__main__":
    main()
