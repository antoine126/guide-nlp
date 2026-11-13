"""TF-IDF from scratch, checked against scikit-learn (chapter 12).

Usage: uv run --group classic python -m guide_nlp.chapter12.tfidf
"""

import math
import re
from collections import Counter

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from guide_nlp.corpus.load import load_articles


def words(text: str) -> list[str]:
    return re.findall(r"[^\W\d_]+", text.lower())


def tfidf_matrix(texts: list[str]) -> tuple[sp.csr_matrix, dict[str, int], np.ndarray]:
    """TF-IDF with smoothed IDF and L2 normalisation, from scratch."""
    docs = [Counter(words(text)) for text in texts]
    df = Counter(term for doc in docs for term in doc)
    vocabulary = {term: i for i, term in enumerate(sorted(df))}
    n_docs = len(docs)
    idf = np.array([math.log((1 + n_docs) / (1 + df[t])) + 1 for t in sorted(df)])
    rows, cols, values = [], [], []
    for i, doc in enumerate(docs):
        for term, tf in doc.items():
            j = vocabulary[term]
            rows.append(i)
            cols.append(j)
            values.append(tf * idf[j])
    X = sp.csr_matrix((values, (rows, cols)), shape=(n_docs, len(vocabulary)))
    norms = np.sqrt(X.multiply(X).sum(axis=1)).A1
    return sp.csr_matrix(sp.diags(1 / norms) @ X), vocabulary, idf


def top_terms(vectorizer: TfidfVectorizer, X: sp.csr_matrix, index: int, k: int = 10) -> list[str]:
    """The k terms with the highest TF-IDF weight in one document."""
    terms = vectorizer.get_feature_names_out()
    row = X[index].tocoo()
    top = sorted(zip(row.data, row.col, strict=True), reverse=True)[:k]
    return [terms[j] for _, j in top]


def main() -> None:
    articles = load_articles()
    texts = [a.text for a in articles]
    X0, vocabulary, _ = tfidf_matrix(texts)
    sk = TfidfVectorizer(token_pattern=r"[^\W\d_]+")
    X1 = sk.fit_transform(texts)
    order = [sk.vocabulary_[w] for w in sorted(vocabulary)]
    print(
        f"same shape: {X0.shape == X1.shape}, "
        f"max abs difference: {abs(X0 - X1[:, order]).max():.1e}"
    )

    vectorizer = TfidfVectorizer(
        token_pattern=r"[^\W\d_]+",  # default ignores one-letter tokens
        sublinear_tf=True,  # default False: raw counts
        smooth_idf=True,  # default True: ln((1+N)/(1+df)) + 1
        norm="l2",  # default "l2"
        min_df=2,  # default 1
    )
    train = [a for a in articles if a.split == "train"]
    test = [a for a in articles if a.split == "test"]
    X_train = vectorizer.fit_transform([a.text for a in train])
    X_test = vectorizer.transform([a.text for a in test])  # never fit on the test set
    print(X_train.shape, X_test.shape)

    index = next(i for i, a in enumerate(articles) if a.title == "Inflation par la demande")
    print(articles[index].title, "->", top_terms(sk, X1, index))


if __name__ == "__main__":
    main()
