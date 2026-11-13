"""Communities of the graph with the Leiden algorithm (chapter 38)."""

import igraph as ig
import leidenalg as la
import networkx as nx


def leiden(graph: nx.Graph, resolution: float = 1.0) -> list[list[str]]:
    nodes = list(graph.nodes())
    position = {n: i for i, n in enumerate(nodes)}
    g = ig.Graph(n=len(nodes), edges=[(position[a], position[b]) for a, b in graph.edges()])
    partition = la.find_partition(
        g,
        la.RBConfigurationVertexPartition,
        weights=[graph[a][b]["weight"] for a, b in graph.edges()],
        resolution_parameter=resolution,
        seed=0,
    )
    return [[nodes[i] for i in community] for community in partition]


if __name__ == "__main__":
    graph = nx.karate_club_graph()
    nx.set_edge_attributes(graph, 1, "weight")
    print([len(c) for c in leiden(graph)])
