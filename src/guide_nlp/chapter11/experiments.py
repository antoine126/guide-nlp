"""The measurements of chapter 11: Zipf, Heaps, sparsity, n-grams, hashing.

Usage: uv run --group classic python -m guide_nlp.chapter11.experiments
"""

from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import murmurhash3_32

from guide_nlp.corpus.load import load_articles, words


def zipf_and_heaps(texts: list[str]) -> None:
    tokens = [t for text in texts for t in words(text)]
    counts = Counter(tokens)
    n, v = len(tokens), len(counts)
    freqs = np.array(sorted(counts.values(), reverse=True), dtype=float)
    hapax = int((freqs == 1).sum())
    print(f"tokens={n} types={v} hapax={hapax} ({hapax / v:.1%})")
    for top in (10, 100, 1000, 10000):
        print(f"  the {top} most frequent words cover {freqs[:top].sum() / n:.1%} of tokens")
    ranks = np.arange(10, 10001)
    slope, intercept = np.polyfit(np.log(ranks), np.log(freqs[ranks - 1]), 1)
    print(f"Zipf fit on ranks 10-10000: s={-slope:.3f} C={np.exp(intercept):.0f}")

    seen: set[str] = set()
    marks = [int(x) for x in np.logspace(3, np.log10(n), 16)]
    curve, next_mark = [], 0
    for i, token in enumerate(tokens, start=1):
        seen.add(token)
        if next_mark < len(marks) and i >= marks[next_mark]:
            curve.append((i, len(seen)))
            next_mark += 1
    curve.append((n, len(seen)))
    sizes, vocab = np.array(curve).T
    beta, log_k = np.polyfit(np.log(sizes), np.log(vocab), 1)
    print(f"Heaps fit: beta={beta:.3f} K={np.exp(log_k):.1f}")


def sparsity(texts: list[str]) -> None:
    X = CountVectorizer().fit_transform(texts)
    n, m = X.shape
    csr_mb = (X.data.nbytes + X.indices.nbytes + X.indptr.nbytes) / 1e6
    print(
        f"CountVectorizer: shape={X.shape} density={X.nnz / (n * m):.4%} "
        f"dense={n * m * 8 / 1e9:.2f} GB csr={csr_mb:.1f} MB"
    )
    variants: dict[str, dict] = {
        "unigrams": {},
        "uni+bi": {"ngram_range": (1, 2)},
        "uni+bi+tri": {"ngram_range": (1, 3)},
        "uni min_df=5": {"min_df": 5},
        "uni+bi min_df=5": {"ngram_range": (1, 2), "min_df": 5},
        "char 3-5 wb": {"analyzer": "char_wb", "ngram_range": (3, 5)},
    }
    for name, kwargs in variants.items():
        Y = CountVectorizer(**kwargs).fit_transform(texts)
        mb = (Y.data.nbytes + Y.indices.nbytes + Y.indptr.nbytes) / 1e6
        print(f"  {name:<18} vocab={Y.shape[1]:>9} nnz={Y.nnz:>10} csr={mb:6.1f} MB")


def hashing_collisions(texts: list[str]) -> None:
    vocabulary = list(CountVectorizer().fit(texts).vocabulary_)
    for bits in (14, 18, 20, 22):
        size = 2**bits
        buckets = Counter(murmurhash3_32(t, positive=True) % size for t in vocabulary)
        colliding = sum(c for c in buckets.values() if c > 1) / len(vocabulary)
        theory = 1 - (1 - 1 / size) ** (len(vocabulary) - 1)
        print(f"  2^{bits} buckets: {colliding:.1%} of terms collide (theory {theory:.1%})")


def main() -> None:
    texts = [a.text for a in load_articles("train")]
    print(f"{len(texts)} training articles")
    zipf_and_heaps(texts)
    sparsity(texts)
    hashing_collisions(texts)


if __name__ == "__main__":
    main()
