"""ROUGE with a French tokenizer, and BERTScore without IDF (chapter 26).

Usage: uv run --group neural python -m guide_nlp.chapter26.metrics
"""

import re

import torch
from rouge_score import rouge_scorer
from transformers import AutoModel, AutoTokenizer

from guide_nlp.config import settings
from guide_nlp.runtime import resolve_device

WORD = re.compile(r"[^\W\d_]+|\d+")
device = resolve_device(settings.device)
tok = AutoTokenizer.from_pretrained("almanach/camembert-base")
bert = AutoModel.from_pretrained("almanach/camembert-base").to(device).eval()


class FrenchTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return WORD.findall(text.lower())


scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], tokenizer=FrenchTokenizer())


@torch.no_grad()
def token_vectors(text: str) -> torch.Tensor:
    enc = tok(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    hidden = bert(**enc).last_hidden_state[0][1:-1]  # without <s> and </s>
    return torch.nn.functional.normalize(hidden, dim=-1)


def bert_score(candidate: str, reference: str) -> tuple[float, float, float]:
    """Zhang et al. (2020), without IDF: greedy matching by cosine similarity."""
    sim = token_vectors(candidate) @ token_vectors(reference).T
    precision = sim.max(dim=1).values.mean().item()
    recall = sim.max(dim=0).values.mean().item()
    return precision, recall, 2 * precision * recall / (precision + recall)


def main() -> None:
    reference = (
        "Le contrôle et la réglementation des marchés financiers visent au bon "
        "fonctionnement des bourses."
    )
    candidate = "La surveillance des marchés financiers vise le bon fonctionnement des bourses."
    for name, score in scorer.score(reference, candidate).items():
        print(f"{name}: F={score.fmeasure:.3f}")
    print("BERTScore P, R, F:", [round(x, 3) for x in bert_score(candidate, reference)])


if __name__ == "__main__":
    main()
