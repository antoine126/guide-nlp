"""The RAG system under test in chapter 41: retrieve, then answer with citations."""

from dataclasses import dataclass
from functools import cache

from guide_nlp.rag import RagBase


@dataclass(frozen=True)
class Chunk:
    source: int
    text: str


@cache
def _rag() -> RagBase:
    return RagBase()


def retrieve(question: str, k: int = 5) -> list[Chunk]:
    rag = _rag()
    return [Chunk(int(rag.source[i]), rag.chunks[i]) for i in rag.hybrid(question, k)]


def answer(question: str, k: int = 5) -> str:
    rag = _rag()
    return rag.ask(question, rag.hybrid(question, k)).content
