"""Training sentences and pretrained vectors for chapters 17 and 18."""

from collections import Counter
from functools import cache
from pathlib import Path

from gensim.models import KeyedVectors

from guide_nlp.chapter14.data import sentences
from guide_nlp.config import settings
from guide_nlp.corpus.load import load_articles

# frWac (Fauconnier, 2015): 155,562 words, 500 dimensions, skip-gram trained on
# 1.6 billion words of French web pages. Download it once:
#   https://embeddings.net/embeddings/frWac_non_lem_no_postag_no_phrase_500_skip_cut100.bin
# and save it as data/raw/pretrained/frwac.bin (the book's copy weighs 312,726,847 bytes).
FRWAC = settings.raw_dir / "pretrained" / "frwac.bin"


@cache
def train_sentences() -> list[list[str]]:
    """The sentences of the training articles, as lists of lowercased words."""
    return list(sentences(load_articles("train")))


def word_counts() -> Counter[str]:
    return Counter(t for s in train_sentences() for t in s)


@cache
def load_frwac(path: Path = FRWAC) -> KeyedVectors:
    load = KeyedVectors.load_word2vec_format
    return load(str(path), binary=True, unicode_errors="ignore")
