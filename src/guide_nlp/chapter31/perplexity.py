"""Perplexity and bits per character of causal models on our test articles (chapter 31).

Usage: uv run --group neural python -m guide_nlp.chapter31.perplexity [MODEL...]
Bits per character compare models whose tokenizers differ; perplexity does not.
"""

import math
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from guide_nlp.config import settings
from guide_nlp.corpus.load import load_articles
from guide_nlp.runtime import resolve_device


@torch.no_grad()
def perplexity(
    tok: AutoTokenizer, model: PreTrainedModel, paragraphs: list[str]
) -> tuple[float, float]:
    total, n_tokens, n_chars = 0.0, 0, 0
    for text in paragraphs:
        enc = tok(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        ids = enc["input_ids"]
        logits = model(**enc).logits
        total += F.cross_entropy(logits[0, :-1].float(), ids[0, 1:], reduction="sum").item()
        n_tokens += ids.shape[1] - 1
        n_chars += len(tok.decode(ids[0, 1:], skip_special_tokens=True))
    return math.exp(total / n_tokens), total / math.log(2) / n_chars  # per token, bits/char


def main(names: list[str]) -> None:
    paragraphs = [
        p for a in load_articles("test")[:60] for p in a.text.split("\n\n") if len(p.split()) > 40
    ][:200]
    device = resolve_device(settings.device)
    for name in names:
        tok = AutoTokenizer.from_pretrained(name)
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16).to(device).eval()
        ppl, bpc = perplexity(tok, model, paragraphs)
        print(f"{name:<36} perplexity={ppl:7.2f} bits/char={bpc:.3f}")


if __name__ == "__main__":
    main(
        sys.argv[1:] or ["openai-community/gpt2", "HuggingFaceTB/SmolLM2-1.7B", "Qwen/Qwen2.5-1.5B"]
    )
