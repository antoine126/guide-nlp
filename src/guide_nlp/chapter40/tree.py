"""A report's table of contents as a tree of summarised nodes (chapter 40)."""

from collections.abc import Callable
from typing import Any

Node = dict[str, Any]


def build_tree(sections: list[dict], summarize: Callable[[dict], str]) -> list[Node]:
    nodes: list[Node] = []
    stack: list[Node] = []
    for s in sections:
        node = dict(
            id=len(nodes),
            title=s["title"],
            level=s["level"],
            page=s["page"],
            start=s["start"],
            end=s["end"],
            children=[],
            parent=None,
            summary=summarize(s),
        )
        while stack and stack[-1]["level"] >= s["level"]:
            stack.pop()
        if stack:
            node["parent"] = stack[-1]["id"]
            stack[-1]["children"].append(node["id"])
        stack.append(node)
        nodes.append(node)
    return nodes


def listing(nodes: list[Node], ids: list[int] | None = None) -> str:
    """Numbered titles and summaries of the given nodes (the roots by default)."""
    chosen = ids if ids is not None else [n["id"] for n in nodes if n["parent"] is None]
    return "\n".join(f"{i}. {nodes[i]['title']} — {nodes[i]['summary']}" for i in chosen)
