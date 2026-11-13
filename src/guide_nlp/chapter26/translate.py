"""Machine translation with NLLB-200 (chapter 26).

Usage: uv run --group neural python -m guide_nlp.chapter26.translate
"""

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from guide_nlp.config import settings
from guide_nlp.runtime import resolve_device

NAME = "facebook/nllb-200-distilled-600M"
device = resolve_device(settings.device)
tok = AutoTokenizer.from_pretrained(NAME, src_lang="fra_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(NAME).to(device).eval()
target = tok.convert_tokens_to_ids("eng_Latn")


def translate(batch: list[str]) -> list[str]:
    enc = tok(batch, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        out = model.generate(**enc, forced_bos_token_id=target, num_beams=5, max_new_tokens=64)
    return tok.batch_decode(out, skip_special_tokens=True)


if __name__ == "__main__":
    print(translate(["Il a acheté une nouvelle voiture rouge.", "Tom finira par tout me dire."]))
