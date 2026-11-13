"""A configurable preprocessing pipeline (chapter 6)."""

from pydantic import BaseModel
from spacy.language import Language

NEGATIONS = {"ne", "n'", "pas", "plus", "jamais", "rien", "aucun",
             "aucune", "sans", "ni"}  # fmt: skip


class PreprocessConfig(BaseModel):
    """Every preprocessing decision, stated once and versioned."""

    lowercase: bool = True
    remove_punctuation: bool = True
    remove_stopwords: bool = True
    keep_negations: bool = True
    lemmatize: bool = True


def preprocess(text: str, nlp: Language, config: PreprocessConfig) -> list[str]:
    """Turn a raw text into a list of normalised tokens."""
    tokens = []
    for tok in nlp(text):
        if tok.is_space or (config.remove_punctuation and tok.is_punct):
            continue
        is_negation = tok.lower_ in NEGATIONS
        if config.remove_stopwords and tok.is_stop and not (config.keep_negations and is_negation):
            continue
        form = tok.lemma_ if config.lemmatize else tok.text
        tokens.append(form.lower() if config.lowercase else form)
    return tokens
