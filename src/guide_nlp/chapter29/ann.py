"""Approximate nearest neighbours with FAISS: HNSW and IVF against exact search (chapter 29).

Usage: uv run --group neural python -m guide_nlp.chapter29.ann
The corpus has only 2,923 articles; the demo indexes 200,000 random unit
vectors of the same dimension to make the trade-offs visible.
"""

import time

import faiss
import numpy as np


def recall_at_10(found: np.ndarray, truth: np.ndarray) -> float:
    return sum(len(set(a) & set(b)) / 10 for a, b in zip(found, truth, strict=True)) / len(truth)


def main(n: int = 200_000, dim: int = 1024, n_queries: int = 200) -> None:
    rng = np.random.default_rng(0)
    vectors = rng.standard_normal((n, dim), dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    queries = vectors[:n_queries] + 0.5 * rng.standard_normal((n_queries, dim), dtype=np.float32)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    exact = faiss.IndexFlatIP(dim)
    exact.add(vectors)

    hnsw = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
    hnsw.hnsw.efConstruction = 200
    hnsw.add(vectors)
    hnsw.hnsw.efSearch = 128

    ivf = faiss.IndexIVFFlat(faiss.IndexFlatIP(dim), dim, 919, faiss.METRIC_INNER_PRODUCT)
    ivf.train(vectors)
    ivf.add(vectors)
    ivf.nprobe = 64

    _, truth = exact.search(queries, 10)
    for name, index in (("exact", exact), ("HNSW", hnsw), ("IVF", ivf)):
        start = time.perf_counter()
        _, found = index.search(queries, 10)
        ms = (time.perf_counter() - start) / n_queries * 1000
        print(f"{name:<6} recall@10={recall_at_10(found, truth):.3f} {ms:.2f} ms/query")


if __name__ == "__main__":
    main()
