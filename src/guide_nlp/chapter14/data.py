"""Sentences of the 'Articles' corpus, shared by the experiments of chapter 14."""

import re
from collections.abc import Iterator

from spacy.lang.fr.stop_words import STOP_WORDS

from guide_nlp.corpus.load import WikiArticle, words

SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
EXTRA_STOP = {
    "l",
    "d",
    "qu",
    "n",
    "s",
    "c",
    "j",
    "m",
    "t",
    "y",
    "a",
    "plus",
    "ainsi",
    "également",
    "aussi",
}
STOP = set(STOP_WORDS) | EXTRA_STOP


def sentences(articles: list[WikiArticle]) -> Iterator[list[str]]:
    """One list of lowercased words per sentence."""
    for article in articles:
        for sentence in SENTENCE_END.split(article.text):
            tokens = words(sentence)
            if tokens:
                yield tokens


def content_words(text: str) -> list[str]:
    """Words without stop words (topic models, LSA)."""
    return [w for w in words(text) if w not in STOP]
