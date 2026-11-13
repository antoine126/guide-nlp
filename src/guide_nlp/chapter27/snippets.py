"""The short examples of chapter 27: a cross-encoder, and unsupervised SimCSE pairs.

Usage: uv run --group neural python -m guide_nlp.chapter27.snippets
"""

from datasets import Dataset
from sentence_transformers import CrossEncoder, SentenceTransformer, models


def cross_encoder() -> None:
    """A cross-encoder reads the query and the document together: precise, slow."""
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)
    query = "satellite qui surveille la montée des eaux"
    candidates = [
        "Le satellite Argos surveille le niveau des océans depuis son orbite.",
        "Le lanceur a décollé de Kourou à 10 h 15.",
        "La banque centrale a relevé ses taux.",
    ]
    scores = reranker.predict([(query, document) for document in candidates])
    for score, document in sorted(zip(scores.tolist(), candidates, strict=True), reverse=True):
        print(f"{score:.3f}  {document}")


def simcse_pairs(sentences: list[str]) -> tuple[SentenceTransformer, Dataset]:
    """SimCSE: the same text twice; dropout makes the two encodings differ."""
    word = models.Transformer("almanach/camembert-base", max_seq_length=64)
    pooling = models.Pooling(word.get_word_embedding_dimension(), pooling_mode="mean")
    model = SentenceTransformer(modules=[word, pooling])
    # the same text twice
    pairs = Dataset.from_dict({"anchor": sentences, "positive": sentences})
    return model, pairs


def main() -> None:
    cross_encoder()
    model, pairs = simcse_pairs(["Le chat dort.", "La Terre tourne autour du Soleil."])
    print(model, pairs)


if __name__ == "__main__":
    main()
