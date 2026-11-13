"""The bag of words, by hand and with scikit-learn (chapter 11).

Usage: uv run --group classic python -m guide_nlp.chapter11.bag_of_words
"""

import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer

DOCS = ["le chien mord l'homme", "l'homme mord le chien", "le chat dort"]


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def by_hand(docs: list[str]) -> np.ndarray:
    vocabulary = sorted({word for doc in docs for word in tokenize(doc)})
    index = {word: i for i, word in enumerate(vocabulary)}
    matrix = np.zeros((len(docs), len(vocabulary)), dtype=int)
    for row, doc in enumerate(docs):
        for word, count in Counter(tokenize(doc)).items():
            matrix[row, index[word]] = count
    return matrix


def main() -> None:
    matrix = by_hand(DOCS)

    # With scikit-learn (the default pattern ignores one-letter words: override it)
    vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
    X = vectorizer.fit_transform(DOCS)
    print(vectorizer.get_feature_names_out().tolist())
    # ['chat', 'chien', 'dort', 'homme', 'l', 'le', 'mord']
    print(X.toarray().tolist())
    # [[0, 1, 0, 1, 1, 1, 1], [0, 1, 0, 1, 1, 1, 1], [1, 0, 1, 0, 0, 1, 0]]
    print((X.toarray() == matrix).all())  # True

    # The hashing trick: no fit, nothing to learn, no vocabulary.
    hashing = HashingVectorizer(n_features=2**20, alternate_sign=True, norm=None)
    H = hashing.transform(DOCS)
    print(H.shape, H.nnz)  # (3, 1048576) 11


if __name__ == "__main__":
    main()
