"""Personalised PageRank from the entities of the question, as in HippoRAG (chapter 38)."""

from collections import defaultdict

import networkx as nx


def ppr_chunks(
    graph: nx.Graph, seeds: list[str], node_chunks: dict[str, set[int]], k: int = 5
) -> list[int]:
    rank = nx.pagerank(graph, alpha=0.5, personalization={n: 1.0 for n in seeds}, weight="weight")
    scores: defaultdict[int, float] = defaultdict(float)
    for node, value in rank.items():
        linked = node_chunks.get(node, set())
        for chunk in linked:
            scores[chunk] += value / max(len(linked), 1) ** 0.5
    return sorted(scores, key=scores.__getitem__, reverse=True)[:k]


if __name__ == "__main__":
    from guide_nlp.chapter38.graph import build, node_chunks

    graph = build(
        [
            ("bardellone bonacossi", "fils de", "pinamonte bonacossi", 0),
            ("pinamonte bonacossi", "mort à", "mantoue", 1),
            ("mantoue", "situé en", "italie", 2),
        ]
    )
    print(ppr_chunks(graph, ["bardellone bonacossi"], node_chunks(graph), k=3))
