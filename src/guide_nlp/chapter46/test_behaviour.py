"""Behavioural tests of a classifier: invariance to harmless edits (chapter 46).

Usage: uv run --group classic pytest src/guide_nlp/chapter46/test_behaviour.py
"""

import random
from collections.abc import Callable

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import LinearSVC

from guide_nlp.chapter15.baselines import TFIDF
from guide_nlp.chapter15.data import rubric_split


def add_typos(text: str, rate: float = 0.02, seed: int = 0) -> str:
    """Swap two adjacent letters in a share of the words."""
    rng = random.Random(seed)
    out = []
    for word in text.split():
        if len(word) > 3 and rng.random() < rate:
            i = rng.randrange(len(word) - 1)
            word = word[:i] + word[i + 1] + word[i] + word[i + 2 :]
        out.append(word)
    return " ".join(out)


@pytest.fixture(scope="module")
def model() -> Pipeline:
    train_texts, train_labels, _, _ = rubric_split()
    return make_pipeline(TfidfVectorizer(**TFIDF), LinearSVC()).fit(train_texts, train_labels)


@pytest.fixture(scope="module")
def samples() -> list[str]:
    return rubric_split()[2][:200]


@pytest.mark.parametrize(
    "transform, tolerated",
    [
        (lambda t: "Cet article a été mis à jour récemment. " + t, 0.02),
        (str.upper, 0.05),
        (lambda t: add_typos(t, rate=0.02), 0.05),
    ],
)
def test_invariance(
    model: Pipeline, samples: list[str], transform: Callable[[str], str], tolerated: float
) -> None:
    before = model.predict(samples)
    after = model.predict([transform(t) for t in samples])
    assert np.mean(before != after) <= tolerated
