"""Word vectors from co-occurrences: PPMI, then truncated SVD (chapter 14).

Usage: uv run --group classic python -m guide_nlp.chapter14.ppmi
"""

from collections import Counter

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from guide_nlp.chapter14.data import sentences
from guide_nlp.corpus.load import load_articles

PROBES = ["planète", "roi", "virus", "banque", "football", "ordinateur", "guerre", "maladie"]


def ppmi_matrix(C: sp.csr_matrix) -> sp.csr_matrix:
    total = C.sum()
    p_word = np.asarray(C.sum(axis=1)).ravel() / total
    coo = C.tocoo()
    pmi = np.log((coo.data / total) / (p_word[coo.row] * p_word[coo.col]))
    keep = pmi > 0
    return sp.csr_matrix((pmi[keep], (coo.row[keep], coo.col[keep])), shape=C.shape)


def cooccurrence_matrix(
    sents: list[list[str]], n_words: int = 20_000, window: int = 5
) -> tuple[sp.csr_matrix, list[str]]:
    """Symmetric counts of the most frequent words within a window."""
    counts = Counter(t for s in sents for t in s)
    vocabulary = [w for w, _ in counts.most_common(n_words)]
    index = {w: i for i, w in enumerate(vocabulary)}
    pairs: Counter[tuple[int, int]] = Counter()
    for s in sents:
        ids = [index.get(w, -1) for w in s]
        for i, a in enumerate(ids):
            if a < 0:
                continue
            for b in ids[i + 1 : i + 1 + window]:
                if b >= 0:
                    pairs[(a, b)] += 1
                    pairs[(b, a)] += 1
    rows, cols, values = zip(*((a, b, c) for (a, b), c in pairs.items()), strict=True)
    C = sp.csr_matrix((np.array(values, dtype=float), (rows, cols)), shape=(n_words, n_words))
    return C, vocabulary


def main() -> None:
    cooccurrences, vocabulary = cooccurrence_matrix(list(sentences(load_articles("train"))))
    index = {w: i for i, w in enumerate(vocabulary)}

    ppmi = ppmi_matrix(cooccurrences)
    svd = TruncatedSVD(n_components=200, random_state=0)
    word_vectors = normalize(svd.fit_transform(ppmi))
    for word in PROBES:
        neighbours = np.argsort(-(word_vectors @ word_vectors[index[word]]))[1:9]
        print(f"{word:<11} -> {[vocabulary[i] for i in neighbours]}")
    # planète     -> ['jupiter', 'uranus', 'neptune', 'pluton', 'naine', 'troyens', ...]


if __name__ == "__main__":
    main()
