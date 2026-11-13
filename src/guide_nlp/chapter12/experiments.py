"""The measurements of chapter 12: IDF values and the preprocessing variants.

Usage: uv run --group classic python -m guide_nlp.chapter12.experiments
"""

import math
import time
from collections import Counter
from collections.abc import Callable

import numpy as np
import scipy.sparse as sp
import snowballstemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from spacy.lang.fr.stop_words import STOP_WORDS

from guide_nlp.chapter13.metrics import mean_metrics, rank_metrics
from guide_nlp.corpus.load import load_articles, load_queries, words

STEMMER = snowballstemmer.stemmer("french")


def no_stop(text: str) -> list[str]:
    return [w for w in words(text) if w not in STOP_WORDS]


def stems(text: str) -> list[str]:
    return STEMMER.stemWords(words(text))


def stems_no_stop(text: str) -> list[str]:
    return STEMMER.stemWords(no_stop(text))


def variants() -> dict[str, CountVectorizer]:
    def tfidf(tokenizer: Callable[[str], list[str]], **kw: bool) -> TfidfVectorizer:
        return TfidfVectorizer(tokenizer=tokenizer, token_pattern=None, **kw)

    return {
        "comptages bruts (cosinus)": CountVectorizer(tokenizer=words, token_pattern=None),
        "TF-IDF standard": tfidf(words),
        "TF-IDF sublinéaire": tfidf(words, sublinear_tf=True),
        "+ mots vides retirés": tfidf(no_stop, sublinear_tf=True),
        "+ racinisation": tfidf(stems, sublinear_tf=True),
        "+ mots vides + racinisation": tfidf(stems_no_stop, sublinear_tf=True),
    }


def l2(X: sp.csr_matrix) -> sp.csr_matrix:
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    return sp.csr_matrix(sp.diags(1 / np.maximum(norms, 1e-12)) @ X)


def main() -> None:
    articles, queries = load_articles(), load_queries()
    texts, ids = [a.text for a in articles], [a.pageid for a in articles]

    df = Counter(term for text in texts for term in set(words(text)))
    for word in ["de", "année", "planète", "galaxie", "virus", "photosynthèse"]:
        idf = math.log((1 + len(texts)) / (1 + df[word])) + 1
        print(f"df({word})={df[word]} idf={idf:.3f}")

    for name, vectorizer in variants().items():
        D = l2(vectorizer.fit_transform(texts))
        rows = []
        start = time.perf_counter()
        for q in queries:
            scores = (D @ l2(vectorizer.transform([q.query])).T).toarray().ravel()
            top = np.argsort(-scores)[:100]
            rows.append(rank_metrics([ids[k] for k in top], set(q.relevant)))
        ms = (time.perf_counter() - start) / len(queries) * 1000
        m = mean_metrics(rows)
        print(
            f"{name:<30} MRR={m['mrr']:.3f} R@10={m['recall']:.3f} "
            f"nDCG@10={m['ndcg']:.3f} {ms:.1f} ms/q vocab={D.shape[1]}"
        )


if __name__ == "__main__":
    main()
