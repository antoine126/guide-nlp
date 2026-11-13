"""Detecting drift: PSI on a score, MMD on embeddings (chapter 46).

Usage: uv run --group neural python -m guide_nlp.chapter46.drift
"""

import numpy as np

from guide_nlp.chapter28.vectors import SOLON, corpus_vectors
from guide_nlp.corpus.load import load_articles


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    edges = np.percentile(reference, np.linspace(0, 100, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(reference, edges)[0] / len(reference) + 1e-6
    q = np.histogram(current, edges)[0] / len(current) + 1e-6
    return float(np.sum((q - p) * np.log(q / p)))


def mmd(a: np.ndarray, b: np.ndarray, gamma: float = 5.0) -> float:
    """RBF kernel on normalised vectors: ||x - y||^2 = 2 - 2 x.y"""

    def kernel(s: np.ndarray) -> np.ndarray:
        return np.exp(2 * gamma * (s - 1))

    return float(kernel(a @ a.T).mean() + kernel(b @ b.T).mean() - 2 * kernel(a @ b.T).mean())


def main() -> None:
    articles = load_articles()
    vectors, _ = corpus_vectors(SOLON)
    rubric = np.array([a.rubric for a in articles])
    rng = np.random.default_rng(0)
    reference = vectors[rng.choice(len(vectors), 500, replace=False)]
    same = vectors[rng.choice(len(vectors), 500, replace=False)]
    sport = vectors[rubric == "sport"][:500]
    print(
        f"MMD same distribution {mmd(reference, same):.4f}, only sport {mmd(reference, sport):.4f}"
    )
    lengths = np.array([len(a.text.split()) for a in articles], dtype=float)
    print(
        f"PSI of the length: same {psi(lengths[::2], lengths[1::2]):.3f}, "
        f"halved {psi(lengths, lengths / 2):.3f}"
    )


if __name__ == "__main__":
    main()
