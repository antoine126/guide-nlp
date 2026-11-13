"""Encode the corpus and the queries once per model, and cache the vectors on disk."""

from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from guide_nlp.config import settings
from guide_nlp.corpus.load import load_articles, load_queries
from guide_nlp.runtime import resolve_device


@dataclass(frozen=True)
class EmbeddingModel:
    """A model and the prefixes it expects (see its model card)."""

    name: str
    query_prefix: str = ""
    doc_prefix: str = ""

    @property
    def slug(self) -> str:
        return self.name.replace("/", "__")


SOLON = EmbeddingModel("OrdalieTech/Solon-embeddings-large-0.1", "query : ")
E5_LARGE = EmbeddingModel("intfloat/multilingual-e5-large", "query: ", "passage: ")


def load(model: EmbeddingModel, max_seq_length: int = 256) -> SentenceTransformer:
    encoder = SentenceTransformer(model.name, device=str(resolve_device(settings.device)))
    encoder.max_seq_length = max_seq_length
    return encoder


def corpus_vectors(model: EmbeddingModel) -> tuple[np.ndarray, np.ndarray]:
    """(documents, queries), L2-normalised, cached in data/interim/emb/."""
    folder = settings.interim_dir / "emb"
    folder.mkdir(parents=True, exist_ok=True)
    doc_path, query_path = folder / f"{model.slug}_docs.npy", folder / f"{model.slug}_queries.npy"
    if not doc_path.exists():
        encoder = load(model)
        texts = [model.doc_prefix + a.text for a in load_articles()]
        queries = [model.query_prefix + q.query for q in load_queries()]
        np.save(doc_path, encoder.encode(texts, batch_size=64, normalize_embeddings=True))
        np.save(query_path, encoder.encode(queries, batch_size=64, normalize_embeddings=True))
    return np.load(doc_path), np.load(query_path)
