"""Classifying rubrics with a prompt: zero-shot, few-shot, chain of thought (chapter 33).

Usage: uv run --group llm python -m guide_nlp.chapter33.prompts [STYLE] [N]
STYLE is bare, zero, few or cot; N test articles (every third article by default).
"""

import sys
import unicodedata

from sklearn.metrics import f1_score

from guide_nlp.corpus.load import RUBRICS, load_articles
from guide_nlp.llm import Message, chat

SHOWN = "astronomie, économie, histoire, informatique, médecine, sport"
SYSTEM = "Tu classes des articles d'encyclopédie."
INSTRUCTION = (
    f"Classe cet article dans une seule rubrique parmi : {SHOWN}. "
    "Réponds uniquement par le nom de la rubrique.\n\nArticle : "
)
FEW = [
    (
        "La nébuleuse d'Orion est visible à l'œil nu dans la constellation du même nom.",
        "astronomie",
    ),
    ("La banque centrale a relevé ses taux pour contenir la hausse des prix.", "économie"),
    ("Le roi Louis IX partit en croisade en 1248 avec son armée.", "histoire"),
    ("Le compilateur traduit le code source en instructions machine.", "informatique"),
    ("Le diabète de type 2 se traite par un régime et des médicaments.", "médecine"),
    ("L'équipe a remporté le championnat après une finale très disputée.", "sport"),
]


def rubric_messages(text: str, style: str = "zero") -> list[Message]:
    head = " ".join(text.split()[:200])
    if style == "bare":
        return [{"role": "user", "content": f"{head}\n\nRubrique ?"}]
    if style == "zero":
        return [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": INSTRUCTION + head},
        ]
    if style == "few":
        messages: list[Message] = [{"role": "system", "content": SYSTEM}]
        for example, rubric in FEW:
            messages += [
                {"role": "user", "content": INSTRUCTION + example},
                {"role": "assistant", "content": rubric},
            ]
        return [*messages, {"role": "user", "content": INSTRUCTION + head}]
    if style == "cot":
        return [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": f"Classe cet article dans une seule rubrique parmi : {SHOWN}. Raisonne "
                "d'abord en deux phrases, puis termine par une ligne « Rubrique : <nom> ».\n\n"
                f"Article : {head}",
            },
        ]
    raise ValueError(style)


def parse_rubric(answer: str) -> str | None:
    """The first rubric named in the answer, accents ignored."""
    a = "".join(
        c for c in unicodedata.normalize("NFD", answer.lower()) if unicodedata.category(c) != "Mn"
    )
    if "rubrique :" in a:
        a = a.rsplit("rubrique :", 1)[1]
    found = [(a.find(r), r) for r in RUBRICS if r in a]
    return min(found)[1] if found else None


def main(style: str = "zero", n: int | None = None) -> None:
    test = load_articles("test")[::3][:n]
    max_tokens = 256 if style == "cot" else 16
    predicted = [
        parse_rubric(chat(rubric_messages(a.text, style), max_tokens=max_tokens).content)
        for a in test
    ]
    gold = [a.rubric for a in test]
    macro_f1 = f1_score(gold, [p or "invalide" for p in predicted], labels=RUBRICS, average="macro")
    print(
        f"{style}: {len(test)} articles, macro F1 = {macro_f1:.3f}, "
        f"unparsable = {sum(p is None for p in predicted)}"
    )


if __name__ == "__main__":
    main(
        sys.argv[1] if len(sys.argv) > 1 else "zero",
        int(sys.argv[2]) if len(sys.argv) > 2 else None,
    )
