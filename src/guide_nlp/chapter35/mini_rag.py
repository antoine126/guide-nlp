"""A first RAG system, end to end, in sixty lines (chapter 35).

Usage: uv run --group neural --group llm python -m guide_nlp.chapter35.mini_rag [N]
Answers the 55 questions of qa.jsonl (or the first N) and counts correct answers,
abstentions on the 15 unanswerable questions, and errors.
"""

import sys

import faiss
import httpx
import numpy as np
from sentence_transformers import SentenceTransformer

from guide_nlp.config import settings
from guide_nlp.corpus.load import WikiArticle, load_articles, load_questions
from guide_nlp.rag import abstains, correct

SYSTEM = (
    "Tu réponds uniquement à partir des extraits fournis. Cite chaque "
    "information avec le numéro de son extrait, sous la forme [1]. Si les "
    "extraits ne contiennent pas la réponse, réponds exactement : "
    "« Je ne sais pas. »"
)


def split_words(text: str, size: int = 200, overlap: int = 40) -> list[str]:
    words = text.split()
    stop = max(len(words) - overlap, 1)
    return [" ".join(words[i : i + size]) for i in range(0, stop, size - overlap)]


class MiniRag:
    def __init__(
        self,
        articles: list[WikiArticle],
        embedder: str = "OrdalieTech/Solon-embeddings-large-0.1",
        llm: str = "qwen2.5:7b-instruct-q4_K_M",
    ) -> None:
        self.llm = llm
        self.encoder = SentenceTransformer(embedder)
        self.encoder.max_seq_length = 512
        self.chunks = [f"{a.title} : {piece}" for a in articles for piece in split_words(a.text)]
        vectors = self.encoder.encode(self.chunks, batch_size=64, normalize_embeddings=True)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors.astype(np.float32))

    def retrieve(self, question: str, k: int = 5) -> list[str]:
        vector = self.encoder.encode(["query : " + question], normalize_embeddings=True)
        _, ids = self.index.search(vector.astype(np.float32), k)
        return [self.chunks[i] for i in ids[0]]

    def answer(self, question: str, k: int = 5) -> str:
        found = self.retrieve(question, k)
        context = "\n\n".join(f"[{n}] {c}" for n, c in enumerate(found, start=1))
        reply = httpx.post(
            f"{settings.ollama_url}/api/chat",
            timeout=600,
            json={
                "model": self.llm,
                "stream": False,
                "options": {"temperature": 0},
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"Extraits :\n{context}\n\nQuestion : {question}"},
                ],
            },
        )
        reply.raise_for_status()
        return reply.json()["message"]["content"]


def main(n: int | None = None) -> None:
    rag = MiniRag(load_articles())
    questions = load_questions("qa")[:n]
    right = abstained = wrong = 0
    for q in questions:
        reply = rag.answer(q.question)
        if abstains(reply):
            abstained += 1
        elif q.answerable and correct(q.answers, reply):
            right += 1
        else:
            wrong += 1
    answerable = sum(q.answerable for q in questions)
    print(
        f"{len(questions)} questions ({answerable} answerable): {right} correct, "
        f"{abstained} abstentions, {wrong} wrong or invented"
    )


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
