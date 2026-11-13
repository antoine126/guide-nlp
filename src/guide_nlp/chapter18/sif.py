"""Document vectors: mean of word vectors, and SIF (chapter 18).

Usage: uv run --group classic python -m guide_nlp.chapter18.sif
"""

from typing import Literal

import numpy as np
from gensim.models import FastText, KeyedVectors
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize

from guide_nlp.chapter15.data import rubric_split
from guide_nlp.chapter17.corpus import FRWAC, load_frwac, train_sentences, word_counts
from guide_nlp.chapter17.train_gensim import PARAMS, train
from guide_nlp.corpus.load import words


def document_vectors(
    texts: list[str], kv: KeyedVectors, word_freq: dict[str, float], a: float = 1e-3
) -> np.ndarray:
    out = np.zeros((len(texts), kv.vector_size))
    for i, text in enumerate(texts):
        tokens = [w for w in words(text) if w in kv]
        if tokens:
            weights = np.array([a / (a + word_freq.get(t, 1e-6)) for t in tokens])
            weights = weights[:, None]
            out[i] = (weights * kv[tokens]).sum(0) / len(tokens)
    return out


def remove_first_component(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    u = TruncatedSVD(1, random_state=0).fit(train).components_  # fitted on train only
    return train - train @ u.T @ u, test - test @ u.T @ u


def mean_vectors(texts: list[str], kv: KeyedVectors) -> np.ndarray:
    out = np.zeros((len(texts), kv.vector_size))
    for i, text in enumerate(texts):
        tokens = [w for w in words(text) if w in kv]
        if tokens:
            out[i] = kv[tokens].mean(0)
    return out


def macro_f1(kind: Literal["mean", "sif"], kv: KeyedVectors, freq: dict[str, float]) -> float:
    train_texts, train_labels, test_texts, test_labels = rubric_split()
    if kind == "mean":
        A, B = mean_vectors(train_texts, kv), mean_vectors(test_texts, kv)
    else:
        A, B = remove_first_component(
            document_vectors(train_texts, kv, freq), document_vectors(test_texts, kv, freq)
        )
    classifier = LogisticRegression(max_iter=3000, C=10).fit(normalize(A), train_labels)
    return float(f1_score(test_labels, classifier.predict(normalize(B)), average="macro"))


def main() -> None:
    counts = word_counts()
    total = sum(counts.values())
    freq = {w: c / total for w, c in counts.items()}
    sentences = train_sentences()
    sets = {
        "word2vec, corpus (100d)": train(sentences).wv,
        "fastText, corpus (100d)": FastText(sentences, min_n=3, max_n=6, **PARAMS).wv,
    }
    if FRWAC.exists():
        sets["frWac, pretrained (500d)"] = load_frwac()
    for name, kv in sets.items():
        print(
            f"{name:<26} mean={macro_f1('mean', kv, freq):.3f} SIF={macro_f1('sif', kv, freq):.3f}"
        )
    # Book, table 18.2: 0.902 / 0.934, 0.912 / 0.934, 0.970 / 0.975 (TF-IDF + SVM: 0.962)


if __name__ == "__main__":
    main()
