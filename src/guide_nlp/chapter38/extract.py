"""Extracting entities and relations from chunks with a JSON schema (chapter 38).

Usage: uv run --group llm python -m guide_nlp.chapter38.extract [TITLE]
"""

import sys

from pydantic import BaseModel, ValidationError

from guide_nlp.corpus.load import load_articles
from guide_nlp.llm import chat
from guide_nlp.rag import chunk_words


class Entite(BaseModel):
    nom: str
    type: str


class Relation(BaseModel):
    source: str
    relation: str
    cible: str


class Extraction(BaseModel):
    entites: list[Entite]
    relations: list[Relation]


PROMPT = (
    "Extrais de ce texte les entités importantes et les relations explicites "
    "entre elles. Types d'entité : PERSONNE, ORGANISATION, LIEU, OEUVRE, EVENEMENT, "
    "CONCEPT. Une relation relie deux entités de ta liste : source, relation (un "
    "verbe court), cible. Écris au plus 12 entités et 12 relations. N'invente rien."
    "\n\nTexte :\n"
)


def extract(chunk: str) -> Extraction | None:
    reply = chat(
        [{"role": "user", "content": PROMPT + chunk}],
        fmt=Extraction.model_json_schema(),
        max_tokens=1024,
    )
    try:
        return Extraction.model_validate_json(reply.content)
    except ValidationError:
        return None


def main(title: str = "Bardellone Bonacossi") -> None:
    article = next(a for a in load_articles() if a.title == title)
    for chunk in chunk_words(article.text)[:2]:
        extraction = extract(f"{article.title} : {chunk}")
        if extraction:
            for r in extraction.relations:
                print(f"{r.source} — {r.relation} — {r.cible}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "Bardellone Bonacossi")
