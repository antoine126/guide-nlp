"""Transforming the query: rewriting, multi-query, HyDE (chapter 37).

Usage: uv run --group neural --group llm python -m guide_nlp.chapter37.transforms
"""

import numpy as np

from guide_nlp.rag import RagBase, llm_text, rrf

rag = RagBase()


def llm(prompt: str, temperature: float = 0.0, seed: int = 0) -> str:
    return llm_text(prompt, temperature=temperature, seed=seed)


def rewrite(q: str) -> str:
    return llm(
        f"Reformule cette demande en une question de recherche claire et "
        f"précise, sans ajouter d'information.\n\nDemande : {q}"
    )


def multi_query(q: str, k: int = 5) -> list[int]:
    lines = llm(
        f"Écris 3 reformulations différentes de cette question, pour un moteur "
        f"de recherche, une par ligne.\n\nQuestion : {q}"
    ).splitlines()
    return rrf([rag.dense(x, 30) for x in [q, *lines] if x.strip()])[:k]


def hyde(q: str, k: int = 5, n: int = 3) -> list[int]:
    fakes = [
        llm(
            f"Écris un court passage d'encyclopédie qui répond à cette question, "
            f"même si tu dois supposer les détails.\n\nQuestion : {q}",
            temperature=0.7,
            seed=s,
        )
        for s in range(n)
    ]
    vector = rag.encoder.encode(fakes, normalize_embeddings=True).mean(axis=0, keepdims=True)
    return rag.index.search(vector / np.linalg.norm(vector), k)[1][0].tolist()


def main() -> None:
    question = "La chambre consulaire de Dijon a disparu à quelle date ?"
    print("rewritten:", rewrite(question))
    for name, ids in (
        ("dense", rag.dense(question)),
        ("multi-query", multi_query(question)),
        ("HyDE", hyde(question)),
    ):
        print(f"{name:<12}", [rag.chunks[i][:60] for i in ids[:3]])


if __name__ == "__main__":
    main()
