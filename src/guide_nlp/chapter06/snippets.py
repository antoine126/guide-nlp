"""The short examples of chapter 6, in the order of the text.

Usage: uv run python -m guide_nlp.chapter06.snippets
"""

import simplemma
import spacy
from spacy.language import Language

from guide_nlp.text.preprocess import PreprocessConfig, preprocess


def lemmas(nlp: Language) -> None:
    """Lemmatisation with spaCy (in context) and simplemma (word by word)."""
    doc = nlp("Les chats mangeaient les souris qu'ils avaient attrapées.")
    print([(t.text, t.lemma_, t.pos_) for t in doc][:5])
    # [('Les', 'le', 'DET'), ('chats', 'chat', 'NOUN'),
    #  ('mangeaient', 'manger', 'ADV'), ('les', 'le', 'DET'),
    #  ('souris', 'souris', 'NOUN')]

    print(
        [
            simplemma.lemmatize(w, lang="fr")
            for w in ["mangeaient", "chats", "attrapées", "avaient", "été"]
        ]
    )
    # ['manger', 'chat', 'attraper', 'avoir', 'être']


def configurable_pipeline(nlp: Language) -> None:
    """The pipeline of ``guide_nlp.text.preprocess``, with and without negations."""
    text = "Les chats ne mangeaient pas les souris qu'ils avaient attrapées !"
    print(preprocess(text, nlp, PreprocessConfig()))
    # ['chat', 'ne', 'manger', 'pas', 'souris', 'attraper']
    print(preprocess(text, nlp, PreprocessConfig(keep_negations=False)))
    # ['chat', 'manger', 'souris', 'attraper']


def main() -> None:
    nlp = spacy.load("fr_core_news_sm")
    for demo in (lemmas, configurable_pipeline):
        print(f"--- {demo.__name__}")
        demo(nlp)


if __name__ == "__main__":
    main()
