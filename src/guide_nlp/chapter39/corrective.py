"""Corrective RAG as a LangGraph state machine: retrieve, grade, rewrite, generate (chapter 39).

Usage: uv run --group rag --group neural python -m guide_nlp.chapter39.corrective
"""

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from guide_nlp.llm import chat
from guide_nlp.rag import RagBase

rag = RagBase()
chunks = rag.chunks


def dense(query: str, k: int = 5) -> list[int]:
    return rag.dense(query, k)


def is_useful(question: str, chunk: str) -> bool:
    reply = chat(
        [
            {
                "role": "user",
                "content": f"Question : {question}\n\nExtrait :\n{chunk}\n\n"
                "Cet extrait contient-il une information utile pour répondre à la question ? "
                "Réponds par oui ou par non.",
            }
        ],
        max_tokens=3,
    )
    return reply.content.strip().lower().startswith("oui")


def rewrite_query(question: str, query: str) -> str:
    reply = chat(
        [
            {
                "role": "user",
                "content": "Une recherche pour la question suivante n'a rien donné d'utile. "
                "Écris une autre requête de recherche, plus précise. "
                "Réponds uniquement par la requête."
                f"\n\nQuestion : {question}\nRequête essayée : {query}",
            }
        ],
        max_tokens=60,
    )
    return reply.content.strip()


def answer(question: str, ids: list[int]) -> str:
    return rag.ask(question, ids).content


class State(TypedDict):
    question: str
    query: str
    documents: list[int]
    kept: list[int]
    attempts: int
    answer: str


def retrieve(s: State) -> dict:
    return {"documents": dense(s["query"], 5), "attempts": s["attempts"] + 1}


def grade(s: State) -> dict:
    kept = [i for i in s["documents"] if is_useful(s["question"], chunks[i])]
    return {"kept": s["kept"] + [i for i in kept if i not in s["kept"]]}


def rewrite(s: State) -> dict:
    return {"query": rewrite_query(s["question"], s["query"])}


def generate(s: State) -> dict:
    if not s["kept"]:
        return {"answer": "Je ne sais pas."}
    return {"answer": answer(s["question"], s["kept"])}


def route(s: State) -> str:
    return "generate" if s["kept"] or s["attempts"] >= 2 else "rewrite"


# ty does not yet match a TypedDict against LangGraph's state protocol.
builder = StateGraph(State)  # ty: ignore[invalid-argument-type]
for name, node in [
    ("retrieve", retrieve),
    ("grade", grade),
    ("rewrite", rewrite),
    ("generate", generate),
]:
    builder.add_node(name, node)  # ty: ignore[invalid-argument-type]
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "grade")
builder.add_conditional_edges("grade", route, {"generate": "generate", "rewrite": "rewrite"})
builder.add_edge("rewrite", "retrieve")
builder.add_edge("generate", END)
graph = builder.compile()


if __name__ == "__main__":
    question = "Jusqu'à quelle date a existé la chambre de commerce et d'industrie de Côte-d'Or ?"
    state = graph.invoke(
        {
            "question": question,
            "query": question,
            "documents": [],
            "kept": [],
            "attempts": 0,
            "answer": "",
        }
    )
    print(state["answer"], "| kept:", state["kept"], "| attempts:", state["attempts"])
