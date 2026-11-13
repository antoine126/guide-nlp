"""Rule-based matching and an entity ruler in front of the statistical NER (chapter 7).

Usage: uv run python -m guide_nlp.chapter07.rules
"""

import spacy
from spacy.language import Language
from spacy.matcher import Matcher, PhraseMatcher


def token_patterns(nlp: Language) -> None:
    # Token patterns: "prix" + proper noun + optional "de" + optional noun.
    matcher = Matcher(nlp.vocab)
    matcher.add(
        "PRIZE",
        [
            [
                {"LOWER": "prix"},
                {"POS": "PROPN"},
                {"LOWER": "de", "OP": "?"},
                {"POS": "NOUN", "OP": "?"},
            ]
        ],
        greedy="LONGEST",
    )
    doc = nlp("Marie Curie a reçu le prix Nobel de chimie en 1911.")
    print([doc[start:end].text for _, start, end in matcher(doc)])
    # ['prix Nobel de chimie']

    # Long term lists, case-insensitive.
    phrases = PhraseMatcher(nlp.vocab, attr="LOWER")
    phrases.add(
        "APPLIANCE", [nlp.make_doc(t) for t in ["machine à laver", "lave-vaisselle", "sèche-linge"]]
    )
    doc = nlp("Ma Machine à laver et mon sèche-linge sont en panne.")
    print([doc[start:end].text for _, start, end in phrases(doc)])
    # ['Machine à laver', 'sèche-linge']


def entity_ruler(nlp: Language) -> None:
    print([t.text for t in nlp("Réf. REF-2024-0042 livrée.")])
    # ['Réf', '.', 'REF-2024', '-', '0042', 'livrée', '.']

    ruler = nlp.add_pipe("entity_ruler", before="ner")
    # add_pipe() is annotated as returning a plain callable; it is an EntityRuler here.
    ruler.add_patterns(  # ty: ignore[unresolved-attribute]
        [
            # One pattern per token produced by the tokenizer.
            {
                "label": "REF",
                "pattern": [
                    {"TEXT": {"REGEX": r"^REF-\d{4}$"}},
                    {"ORTH": "-"},
                    {"TEXT": {"REGEX": r"^\d{4}$"}},
                ],
            },
            {"label": "ORG", "pattern": "Datalab"},
        ]
    )
    doc = nlp("La commande REF-2024-0042 a été livrée par Datalab à Grenoble.")
    print([(ent.text, ent.label_) for ent in doc.ents])
    # [('REF-2024-0042', 'REF'), ('Datalab', 'ORG'), ('Grenoble', 'LOC')]


def main() -> None:
    nlp = spacy.load("fr_core_news_sm")
    token_patterns(nlp)
    entity_ruler(nlp)


if __name__ == "__main__":
    main()
