"""Scaled dot-product attention in NumPy, with and without a causal mask (chapter 24).

Usage: uv run python -m guide_nlp.chapter24.attention
"""

import math

import numpy as np

TOKENS = ["le", "chat", "boit", "lait"]
K = np.array([[0.2, 0.1, 0.0], [2.0, 0.2, 1.8], [0.3, 2.2, 0.0], [1.8, 0.1, 0.4]])
Q = np.array([[0.2, 0.1, 1.8], [0.3, 2.0, 0.1], [2.0, 0.1, 1.5], [0.1, 1.9, 0.2]])
V = np.array([[0.0, 0.1], [1.0, 0.0], [0.0, 1.0], [0.8, 0.2]])


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def attention(
    Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Scaled dot-product attention."""
    scores = Q @ K.T / math.sqrt(K.shape[-1])
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    weights = softmax(scores)
    return weights @ V, weights


def main() -> None:
    _, weights = attention(Q, K, V)
    print(weights.round(2))
    # [[0.08 0.67 0.1  0.15]
    #  [0.06 0.11 0.74 0.09]
    #  [0.02 0.77 0.03 0.18]
    #  [0.07 0.11 0.73 0.08]]

    causal = np.tril(np.ones((4, 4), dtype=bool))
    print(attention(Q, K, V, causal)[1].round(2))
    # [[1.   0.   0.   0.  ]
    #  [0.37 0.63 0.   0.  ]
    #  [0.02 0.94 0.03 0.  ]
    #  [0.07 0.11 0.73 0.08]]


if __name__ == "__main__":
    main()
