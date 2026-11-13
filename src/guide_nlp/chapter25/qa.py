"""Extractive question answering: the best span under a length limit (chapter 25).

Usage: uv run --group neural python -m guide_nlp.chapter25.qa
"""

import math

import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

NAME = "etalab-ia/camembert-base-squadFR-fquad-piaf"
tok = AutoTokenizer.from_pretrained(NAME)
model = AutoModelForQuestionAnswering.from_pretrained(NAME).eval()


@torch.no_grad()
def answer(question: str, context: str, max_answer: int = 30) -> tuple[str, float]:
    enc = tok(
        question,
        context,
        truncation="only_second",
        max_length=384,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = enc.pop("offset_mapping")[0].tolist()
    in_context = [s == 1 for s in enc.sequence_ids(0)]
    out = model(**enc)
    start, end = out.start_logits[0].log_softmax(-1), out.end_logits[0].log_softmax(-1)
    best, span = -1e9, (0, 0)
    for i in range(len(offsets)):
        for j in range(i, min(i + max_answer, len(offsets))):
            if in_context[i] and in_context[j] and start[i] + end[j] > best:
                best, span = (start[i] + end[j]).item(), (i, j)
    return context[offsets[span[0]][0] : offsets[span[1]][1]], math.exp(best)


def main() -> None:
    context = (
        "Le lanceur a décollé de Kourou à 10 h 15, heure locale, avec à son bord le "
        "satellite d'observation Argos. Conçu pour surveiller la montée des eaux, "
        "l'engin restera en orbite pendant sept ans."
    )
    for question in (
        "D'où le lanceur a-t-il décollé ?",
        "Combien de temps l'engin restera-t-il en orbite ?",
    ):
        text, probability = answer(question, context)
        print(f"{question} -> {text!r} ({probability:.2f})")


if __name__ == "__main__":
    main()
