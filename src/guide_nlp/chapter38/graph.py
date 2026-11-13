"""From extracted triples to a weighted graph (chapter 38)."""

import re
import unicodedata
from collections.abc import Callable, Iterable

import networkx as nx

Triple = tuple[str, str, str, int]  # source, relation, target, chunk id


def norm(name: str) -> str:
    """Lowercase, no accents, no leading article: one spelling per entity."""
    s = "".join(
        c for c in unicodedata.normalize("NFD", name.lower()) if unicodedata.category(c) != "Mn"
    )
    s = re.sub(r"^(l'|le |la |les |un |une |des |du |de |d')", "", s.strip())
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s'-]", " ", s)).strip()


def build(triples: Iterable[Triple], alias: Callable[[str], str] = lambda x: x) -> nx.Graph:
    """Undirected graph; an edge keeps its weight, its relations and its chunks."""
    graph = nx.Graph()
    for a, relation, b, chunk in triples:
        a, b = alias(a), alias(b)
        if a == b:
            continue
        if graph.has_edge(a, b):
            graph[a][b]["weight"] += 1
            graph[a][b]["rels"].add(relation)
            graph[a][b]["chunks"].add(chunk)
        else:
            graph.add_edge(a, b, weight=1, rels={relation}, chunks={chunk})
    return graph


def node_chunks(graph: nx.Graph) -> dict[str, set[int]]:
    """The chunks in which each node takes part."""
    out: dict[str, set[int]] = {n: set() for n in graph.nodes()}
    for a, b, data in graph.edges(data=True):
        out[a] |= data["chunks"]
        out[b] |= data["chunks"]
    return out
