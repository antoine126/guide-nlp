"""A documentary assistant over two sources: the articles and the reports (chapter 47).

Usage: uv run --group neural --group llm python -m guide_nlp.chapter47.assistant "QUESTION"
"""

import sys
from collections.abc import Sequence

import faiss
import numpy as np

from guide_nlp.chapter36.chunking import recursive
from guide_nlp.corpus.load import load_reports
from guide_nlp.rag import RagBase


def search_both(
    query_vector: np.ndarray, sources: Sequence[tuple[faiss.Index, list[str]]], k: int = 5
) -> list[str]:
    """A single ranking over (index, chunks) pairs: the code decides, not the model."""
    pool = []
    for index, chunks in sources:
        scores, ids = index.search(query_vector, k)
        pool += [(float(s), chunks[i]) for s, i in zip(scores[0], ids[0], strict=True)]
    return [text for _, text in sorted(pool, key=lambda x: -x[0])[:k]]


def report_index(rag: RagBase) -> tuple[faiss.Index, list[str]]:
    chunks = [f"{r.title} : {r.text[a:b]}" for r in load_reports() for a, b in recursive(r, 200)]
    vectors = rag.encoder.encode(chunks, batch_size=64, normalize_embeddings=True)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors.astype(np.float32))
    return index, chunks


def main(question: str) -> None:
    rag = RagBase()
    sources = [(rag.index, rag.chunks), report_index(rag)]
    texts = search_both(rag.encode_query(question), sources)
    print(rag.ask(question, [], texts=texts).content)


if __name__ == "__main__":
    main(
        sys.argv[1]
        if len(sys.argv) > 1
        else "Quel était le taux de chômage en France en 2022 selon l'Insee ?"
    )
