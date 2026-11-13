"""Ranking metrics shared by every search chapter (chapter 13, then 27 to 41)."""

import math
from collections.abc import Hashable, Sequence

import numpy as np


def rank_metrics(ranked_ids: Sequence[Hashable], relevant: set, k: int = 10) -> dict[str, float]:
    """Reciprocal rank, recall@k and nDCG@k of one ranked list (binary relevance)."""
    rr = 0.0
    for position, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant:
            rr = 1.0 / position
            break
    hits = [doc_id in relevant for doc_id in ranked_ids[:k]]
    dcg = sum(hit / math.log2(i + 2) for i, hit in enumerate(hits))
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return {"mrr": rr, "recall": sum(hits) / len(relevant), "ndcg": dcg / idcg}


def mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    """Average each metric over the queries."""
    return {key: sum(r[key] for r in rows) / len(rows) for key in rows[0]}


def evaluate_scores(
    scores: np.ndarray, ids: Sequence[Hashable], relevant: list[set], top: int = 100
) -> list[dict[str, float]]:
    """Per-query metrics from a (queries x documents) score matrix."""
    rows = []
    for row, wanted in zip(scores, relevant, strict=True):
        order = np.argpartition(-row, min(top, len(row) - 1))[:top]
        order = order[np.argsort(-row[order])]
        rows.append(rank_metrics([ids[i] for i in order], wanted))
    return rows


def paired_bootstrap(
    a: list[float], b: list[float], n: int = 10_000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean difference a - b over queries, with a 95 % bootstrap interval."""
    rng = np.random.default_rng(seed)
    d = np.asarray(a) - np.asarray(b)
    means = d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
