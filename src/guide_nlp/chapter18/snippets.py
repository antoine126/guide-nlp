"""The short examples of chapter 18: FastText and unseen words, a gender axis.

Usage: uv run --group classic python -m guide_nlp.chapter18.snippets
"""

import numpy as np
from gensim.models import FastText, KeyedVectors

from guide_nlp.chapter17.corpus import FRWAC, load_frwac, train_sentences

OCCUPATIONS = [
    "secrétaire",
    "pilote",
    "journaliste",
    "ministre",
    "architecte",
    "dentiste",
    "psychologue",
    "bibliothécaire",
    "juge",
    "médecin",
    "nourrice",
    "ingénieur",
    "soldat",
    "maçon",
    "mannequin",
    "concierge",
    "chirurgien",
    "vendeuse",
    "garagiste",
    "pédiatre",
]


def unseen_words() -> None:
    """FastText builds a vector for a word it never saw, from its character n-grams."""
    model = FastText(
        train_sentences(),
        vector_size=100,
        window=5,
        min_count=5,
        sg=1,
        negative=5,
        sample=1e-4,
        epochs=10,
        seed=0,
        workers=8,
        min_n=3,
        max_n=6,
    )
    print("exoplanétaire" in model.wv.key_to_index)  # False: below min_count
    print(model.wv.most_similar("exoplanétaire", topn=3))


def gender_axis(kv: KeyedVectors, occupations: list[str]) -> list[tuple[str, float]]:
    """Projection of each word on an axis built from three gendered pairs."""
    direction = kv["elle"] - kv["il"] + kv["femme"] - kv["homme"] + kv["mère"] - kv["père"]
    direction /= np.linalg.norm(direction)
    scores = {w: float((kv[w] / np.linalg.norm(kv[w])) @ direction) for w in occupations if w in kv}
    return sorted(scores.items(), key=lambda kv_: kv_[1])


def main() -> None:
    unseen_words()
    if FRWAC.exists():
        print([(w, round(s, 3)) for w, s in gender_axis(load_frwac(), OCCUPATIONS)])


if __name__ == "__main__":
    main()
