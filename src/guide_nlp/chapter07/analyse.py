"""Part of speech, dependencies, morphology and named entities with spaCy (chapter 7).

Usage: uv run python -m guide_nlp.chapter07.analyse
"""

import spacy
from spacy.language import Language


def show_analysis(nlp: Language) -> None:
    doc = nlp("Emmanuel Macron a inauguré mardi le nouveau laboratoire du CNRS à Grenoble.")
    for tok in doc[:5]:
        print(f"{tok.text:<9} {tok.pos_:<5} {tok.dep_:<10} {tok.head.text:<9} {tok.morph}")
    # Emmanuel  PROPN nsubj      inauguré  Gender=Masc|Number=Sing
    # Macron    PROPN flat:name  Emmanuel  Gender=Masc|Number=Sing
    # a         AUX   aux:tense  inauguré  Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin
    # inauguré  VERB  ROOT       inauguré  Gender=Masc|Number=Sing|Tense=Past|VerbForm=Part
    # mardi     NOUN  obl:mod    inauguré  Gender=Masc|Number=Sing


def show_entities(nlp: Language) -> None:
    doc = nlp("Marie Curie a reçu le prix Nobel de chimie en 1911 à Stockholm.")
    print([(ent.text, ent.label_) for ent in doc.ents])
    # [('Marie Curie', 'PER'), ('prix Nobel de chimie', 'MISC'), ('Stockholm', 'LOC')]

    doc = nlp("La société Acme a racheté la start-up lyonnaise Datalab en 2023.")
    print([(ent.text, ent.label_) for ent in doc.ents])
    # [('Acme', 'ORG'), ('Datalab', 'PER')]


def entities_in_batches(texts: list[str]) -> list[list[tuple[str, str]]]:
    """Only what is needed, in batches: often an order of magnitude faster."""
    nlp = spacy.load("fr_core_news_sm", exclude=["parser"])
    return [[(ent.text, ent.label_) for ent in doc.ents] for doc in nlp.pipe(texts, batch_size=256)]


def main() -> None:
    nlp = spacy.load("fr_core_news_sm")
    show_analysis(nlp)
    show_entities(nlp)
    print(entities_in_batches(["Victor Hugo est né à Besançon."]))


if __name__ == "__main__":
    main()
