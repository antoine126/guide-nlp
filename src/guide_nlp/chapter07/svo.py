"""Subject-verb-object triples from the dependency tree (chapter 7).

Usage: uv run python -m guide_nlp.chapter07.svo
"""

import spacy
from spacy.tokens import Doc


def extract_svo(doc: Doc) -> list[tuple[str, str, str]]:
    """Extract (subject, verb lemma, object) triples from a parsed Doc."""
    triples = []
    for tok in doc:
        if tok.dep_ not in ("nsubj", "nsubj:pass"):
            continue
        verb = tok.head
        for obj in (child for child in verb.children if child.dep_ == "obj"):
            subject = doc[tok.left_edge.i : tok.right_edge.i + 1]
            complement = doc[obj.left_edge.i : obj.right_edge.i + 1]
            triples.append((subject.text, verb.lemma_, complement.text))
    return triples


def main() -> None:
    nlp = spacy.load("fr_core_news_sm")
    for text in [
        "Emmanuel Macron a inauguré mardi le nouveau laboratoire du CNRS à Grenoble.",
        "La banque a accordé un prêt à l'entreprise.",
        "Marie Curie a reçu le prix Nobel de chimie en 1911 à Stockholm.",
    ]:
        print(extract_svo(nlp(text)))
    # [('Emmanuel Macron', 'inaugurer', 'le nouveau laboratoire du CNRS à Grenoble')]
    # [('La banque', 'accorder', "un prêt à l'entreprise")]
    # [('Marie Curie', 'recevoir', 'le prix Nobel de chimie')]


if __name__ == "__main__":
    main()
