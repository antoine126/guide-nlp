"""The RAG base shared by chapters 37 to 41.

The 'Articles' corpus cut into windows of 200 words (40 of overlap), each
prefixed by its title; Solon vectors in a FAISS index; BM25 on the same
chunks; and a grounded answer that cites its extracts or abstains. This is
the first system of chapter 35, factored so that later chapters can change
one piece at a time.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from functools import cached_property

import bm25s
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from guide_nlp.chapter28.vectors import SOLON, load
from guide_nlp.config import settings
from guide_nlp.corpus.load import WikiArticle, load_articles, words
from guide_nlp.llm import ChatResult, chat

GROUNDED = (
    "Tu réponds uniquement à partir des extraits fournis. Cite chaque information avec "
    "le numéro de son extrait, sous la forme [1]. Si les extraits ne contiennent pas la "
    "réponse, réponds exactement : « Je ne sais pas. »"
)
ABSTAIN = re.compile(
    r"je ne sais pas|ne (?:contiennent|contient|mentionnent|mentionne|précisent|précise|"
    r"fournissent|fournit|donnent|donne) pas|pas d'information|aucune information",
    re.I,
)


def chunk_words(text: str, size: int = 200, overlap: int = 40) -> list[str]:
    tokens = text.split()
    return [
        " ".join(tokens[i : i + size])
        for i in range(0, max(len(tokens) - overlap, 1), size - overlap)
    ]


def rrf(lists: Sequence[Sequence[int]], k: int | None = None, c: int = 60) -> list[int]:
    """Reciprocal rank fusion of several ranked lists of chunk ids."""
    scores: dict[int, float] = {}
    for ranking in lists:
        for rank, i in enumerate(ranking, 1):
            scores[i] = scores.get(i, 0.0) + 1 / (c + rank)
    fused = sorted(scores, key=scores.__getitem__, reverse=True)
    return fused[:k] if k else fused


@dataclass
class RagBase:
    """Chunks, dense and lexical indexes, and a grounded generator."""

    articles: list[WikiArticle] = field(default_factory=load_articles)
    size: int = 200
    overlap: int = 40

    @cached_property
    def chunks(self) -> list[str]:
        return [
            f"{a.title} : {piece}"
            for a in self.articles
            for piece in chunk_words(a.text, self.size, self.overlap)
        ]

    @cached_property
    def source(self) -> np.ndarray:
        """The pageid of the article each chunk comes from."""
        return np.array(
            [a.pageid for a in self.articles for _ in chunk_words(a.text, self.size, self.overlap)]
        )

    @cached_property
    def encoder(self) -> SentenceTransformer:
        return load(SOLON, max_seq_length=512)

    @cached_property
    def vectors(self) -> np.ndarray:
        path = settings.interim_dir / "emb" / f"rag{self.size}_solon.npy"
        if path.exists():
            vectors = np.load(path)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            vectors = self.encoder.encode(
                self.chunks, batch_size=64, normalize_embeddings=True
            ).astype(np.float32)
            np.save(path, vectors)
        assert len(vectors) == len(self.chunks)
        return vectors

    @cached_property
    def index(self) -> faiss.IndexFlatIP:
        index = faiss.IndexFlatIP(self.vectors.shape[1])
        index.add(self.vectors)
        return index

    @cached_property
    def bm25(self) -> bm25s.BM25:
        retriever = bm25s.BM25(method="lucene")
        retriever.index([words(c) for c in self.chunks], show_progress=False)
        return retriever

    def encode_query(self, question: str) -> np.ndarray:
        return self.encoder.encode(
            [SOLON.query_prefix + question], normalize_embeddings=True
        ).astype(np.float32)

    def dense(self, question: str, k: int = 5) -> list[int]:
        return self.index.search(self.encode_query(question), k)[1][0].tolist()

    def lexical(self, question: str, k: int = 5) -> list[int]:
        return self.bm25.retrieve([words(question)], k=k, show_progress=False)[0][0].tolist()

    def hybrid(self, question: str, k: int = 5, depth: int = 50) -> list[int]:
        return rrf([self.dense(question, depth), self.lexical(question, depth)], k)

    def ask(
        self,
        question: str,
        ids: Sequence[int],
        system: str = GROUNDED,
        max_tokens: int = 200,
        texts: Sequence[str] | None = None,
    ) -> ChatResult:
        """Answer from the numbered extracts only."""
        passages = texts if texts is not None else [self.chunks[i] for i in ids]
        context = "\n\n".join(f"[{n}] {c}" for n, c in enumerate(passages, 1))
        return chat(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Extraits :\n{context}\n\nQuestion : {question}"},
            ],
            max_tokens=max_tokens,
        )

    def part_hit(self, source: int, answers: Sequence[str], ids: Sequence[int]) -> bool:
        """Did the retrieved chunks include the source article with its answer?"""
        return any(
            self.source[i] == source and any(a.lower() in self.chunks[i].lower() for a in answers)
            for i in ids
        )


def llm_text(prompt: str, temperature: float = 0.0, seed: int = 0, max_tokens: int = 200) -> str:
    """A single-turn call, text only."""
    return chat(
        [{"role": "user", "content": prompt}],
        temperature=temperature,
        seed=seed,
        max_tokens=max_tokens,
    ).content.strip()


def abstains(answer: str) -> bool:
    return ABSTAIN.search(answer) is not None


def correct(answers: Sequence[str], text: str) -> bool:
    return any(a.lower() in text.lower() for a in answers)
