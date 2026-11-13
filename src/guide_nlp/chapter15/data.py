"""Texts and rubrics of the 'Articles' corpus, as used by chapter 15."""

from spacy.lang.fr.stop_words import STOP_WORDS

from guide_nlp.corpus.load import load_articles, words


def rubric_split() -> tuple[list[str], list[str], list[str], list[str]]:
    """train_texts, train_labels, test_texts, test_labels."""
    train, test = load_articles("train"), load_articles("test")
    return (
        [a.text for a in train],
        [a.rubric for a in train],
        [a.text for a in test],
        [a.rubric for a in test],
    )


def tokenize(text: str) -> list[str]:
    """Words of at least three letters, without stop words (clustering)."""
    return [w for w in words(text) if w not in STOP_WORDS and len(w) > 2]
