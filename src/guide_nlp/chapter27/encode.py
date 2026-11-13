"""Sentence embeddings with Sentence Transformers (chapter 27).

Usage: uv run --group neural python -m guide_nlp.chapter27.encode
"""

from sentence_transformers import SentenceTransformer


def main() -> None:
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    sentences = [
        "Le chat dort sur le canapé.",
        "Un félin se repose sur le sofa.",
        "La banque centrale relève ses taux.",
    ]
    vectors = model.encode(sentences, normalize_embeddings=True)
    print(vectors.shape)  # (3, 768)
    print(f"{vectors[0] @ vectors[1]:.3f}")  # 0.801
    print(f"{vectors[0] @ vectors[2]:.3f}")  # 0.081


if __name__ == "__main__":
    main()
