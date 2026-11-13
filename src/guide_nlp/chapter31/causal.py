"""Causal masking: a token never sees the following ones (chapter 31).

Usage: uv run --group neural python -m guide_nlp.chapter31.causal
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").eval()
    a = tok("The cat sat on the mat and looked", return_tensors="pt")
    b = tok("The cat sat on the mat and barked", return_tensors="pt")
    with torch.no_grad():
        la, lb = model(**a).logits, model(**b).logits
    n = a["input_ids"].shape[1]
    print(f"{(la[0, : n - 1] - lb[0, : n - 1]).abs().max():.1f}")  # 0.0
    print(f"{(la[0, -1] - lb[0, -1]).abs().max():.1f}")  # 15.0


if __name__ == "__main__":
    main()
