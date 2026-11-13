"""The search tool exposed as an MCP server (chapter 39).

Usage: uv run --group rag --group neural python -m guide_nlp.chapter39.mcp_server
(normally started by an MCP client, through stdio)
"""

from mcp.server import MCPServer

from guide_nlp.rag import RagBase

mcp = MCPServer("recherche-corpus")
rag = RagBase()
by_id = {a.pageid: a for a in rag.articles}


@mcp.tool()
def chercher(requete: str, k: int = 5) -> list[dict]:
    """Cherche dans l'encyclopédie et renvoie les extraits les plus proches de la requête."""
    return [
        {"titre": by_id[int(rag.source[i])].title, "extrait": rag.chunks[i]}
        for i in rag.dense(requete, k)
    ]


if __name__ == "__main__":
    mcp.run()
