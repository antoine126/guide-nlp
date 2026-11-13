"""Contextual word vectors: one vector per occurrence, not per word (chapter 25).

Usage: uv run --group neural python -m guide_nlp.chapter25.context
"""

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

NAME = "almanach/camembert-base"
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModel.from_pretrained(NAME).eval()


@torch.no_grad()
def word_vector(tokens: list[str], index: int, layer: int = -1) -> np.ndarray:
    """Mean of the sub-tokens of one word, in a sentence given as a list of words."""
    enc = tok(tokens, is_split_into_words=True, return_tensors="pt")
    hidden = model(**enc, output_hidden_states=True).hidden_states[layer][0]
    return hidden[[i for i, w in enumerate(enc.word_ids()) if w == index]].mean(0).numpy()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    sentences = [
        "L' avocat a plaidé devant le tribunal .",
        "L' avocat de la défense a demandé un report .",
        "J' ai mangé un avocat bien mûr .",
    ]
    vectors = [word_vector(s.split(), s.split().index("avocat")) for s in sentences]
    print(f"law / law   {cosine(vectors[0], vectors[1]):.3f}")
    print(f"law / fruit {cosine(vectors[0], vectors[2]):.3f}")


if __name__ == "__main__":
    main()
