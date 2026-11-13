"""Faithfulness: the share of an answer's claims that the context supports (chapter 41).

Usage: uv run --group llm python -m guide_nlp.chapter41.faithfulness
"""

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from guide_nlp.llm import chat


class Claims(BaseModel):
    affirmations: list[str]


def faithfulness(answer: str, context: str, llm: Callable[..., str]) -> float | None:
    reply = llm(
        f"Découpe cette réponse en affirmations simples et autonomes.\n\nRéponse : {answer}",
        schema=Claims.model_json_schema(),
    )
    claims = Claims.model_validate_json(reply).affirmations
    if not claims:
        return None
    supported = 0
    for claim in claims:
        verdict = llm(
            f"Contexte :\n{context}\n\nAffirmation : {claim}\n\n"
            "Le contexte appuie-t-il cette affirmation ? "
            "Réponds par oui ou par non."
        )
        supported += verdict.strip().lower().startswith("oui")
    return supported / len(claims)


def llm(prompt: str, schema: dict[str, Any] | None = None) -> str:
    return chat([{"role": "user", "content": prompt}], fmt=schema, max_tokens=300).content


def main() -> None:
    context = (
        "Chambre de commerce et d'industrie de Côte-d'Or : établissement public créé "
        "le 19 mars 2009, dissous le 1er février 2021 lors de sa fusion."
    )
    for answer in (
        "La chambre a existé jusqu'au 1er février 2021.",
        "La chambre a existé jusqu'en 2021 et comptait 300 salariés.",
    ):
        print(f"{faithfulness(answer, context, llm)}  {answer}")


if __name__ == "__main__":
    main()
