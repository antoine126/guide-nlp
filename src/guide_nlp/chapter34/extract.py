"""Invoice extraction with evidence: every value must be quoted from the document (chapter 34).

Usage: uv run --group llm python -m guide_nlp.chapter34.extract
"""

import re
from typing import Any

import httpx
from pydantic import BaseModel

from guide_nlp.config import settings


class Facture(BaseModel):
    numero: str | None
    numero_preuve: str | None
    date: str | None
    date_preuve: str | None
    total: float | None  # a number: the model cannot decorate it
    total_preuve: str | None
    fournisseur: str | None
    fournisseur_preuve: str | None
    ville: str | None
    ville_preuve: str | None


PROMPT = (
    "Extrais les champs de ce document. numero : le numéro de facture. date : "
    "la date au format AAAA-MM-JJ. total : le montant total TTC (nombre). "
    "fournisseur : l'entreprise émettrice. ville : la ville du fournisseur. "
    "Pour chaque champ, le champ _preuve contient la citation exacte du document "
    "qui le justifie. Si l'information est absente, mets le champ et sa preuve "
    "à null. N'invente rien.\n\nDocument :\n"
)
AMOUNT = re.compile(r"\d[\d  \xa0]*(?:[.,]\d{1,2})?")


def extract(text: str) -> Facture:
    reply = httpx.post(
        f"{settings.ollama_url}/api/chat",
        timeout=600,
        json={
            "model": settings.llm_model,
            "stream": False,
            "format": Facture.model_json_schema(),
            "options": {"temperature": 0},
            "messages": [{"role": "user", "content": PROMPT + text}],
        },
    )
    reply.raise_for_status()
    return Facture.model_validate_json(reply.json()["message"]["content"])


def squeeze(text: str) -> str:
    return " ".join(text.split()).lower()


def to_number(text: str) -> float:
    return round(float(re.sub(r"\s", "", text).replace(",", ".")), 2)


def supported(value: Any, quote: str | None, document: str, contains: bool = True) -> bool:
    """The quote must occur in the document, and contain the value."""
    if value in (None, "") or not quote or squeeze(quote) not in squeeze(document):
        return False
    if not contains:  # a date is reformatted: it cannot appear as such
        return True
    if isinstance(value, float):
        return value in [to_number(a) for a in AMOUNT.findall(quote)]
    return squeeze(value) in squeeze(quote)


def main() -> None:
    document = (
        "Bonjour Sophie Girard,\nVeuillez trouver ci-joint la facture FA-2021-2208 de "
        "Plomberie Marchand datée du 26 octobre 2021. Un acompte de 2 665,75 euros a "
        "déjà été versé sur un total de 8 885,84 euros.\nCordialement,\n"
        "Le service comptable, Rennes."
    )
    facture = extract(document)
    for field in ("numero", "date", "total", "fournisseur", "ville"):
        value, quote = getattr(facture, field), getattr(facture, f"{field}_preuve")
        ok = supported(value, quote, document, contains=field != "date")
        print(f"{field:<12} {value!s:<22} {'supported' if ok else 'REJECTED'}  « {quote} »")


if __name__ == "__main__":
    main()
