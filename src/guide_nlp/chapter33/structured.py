"""Structured output: a JSON schema imposed on the decoder, checked by pydantic (chapter 33).

Usage: uv run --group llm python -m guide_nlp.chapter33.structured
"""

import httpx
from pydantic import BaseModel, ValidationError

from guide_nlp.config import settings
from guide_nlp.corpus.load import load_articles

INSTRUCTIONS = (
    "Lis cet article et produis une fiche au format JSON avec les champs : sujet "
    "(chaîne), rubrique (une parmi : astronomie, économie, histoire, informatique, "
    "médecine, sport), annee_principale (entier ou null), mots_cles (liste de "
    "chaînes)."
)


class Fiche(BaseModel):
    sujet: str
    rubrique: str
    annee_principale: int | None
    mots_cles: list[str]


def extract(text: str, retries: int = 2) -> Fiche | None:
    payload = {
        "model": settings.llm_model,
        "stream": False,
        "format": Fiche.model_json_schema(),
        "options": {"temperature": 0},
        "messages": [{"role": "user", "content": INSTRUCTIONS + "\n\nArticle : " + text}],
    }
    for _ in range(retries + 1):
        reply = httpx.post(f"{settings.ollama_url}/api/chat", json=payload, timeout=300).json()
        try:
            return Fiche.model_validate_json(reply["message"]["content"])
        except ValidationError:
            continue
    return None


def main() -> None:
    for article in load_articles("test")[:3]:
        fiche = extract(" ".join(article.text.split()[:200]))
        print(article.title, "->", fiche)


if __name__ == "__main__":
    main()
