"""Fine-tuning BARThez to write the lead of a Wikipedia article (chapter 26).

Usage: uv run --group neural python -m guide_nlp.chapter26.finetune [N_TRAIN]
Each pair is (lead paragraph, rest of the article): the model learns to
summarise the body into its lead.
"""

import re
import sys

import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    PreTrainedTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from guide_nlp.config import settings
from guide_nlp.corpus.load import load_articles

NAME = "moussaKam/barthez"
tok = PreTrainedTokenizerFast.from_pretrained(  # AutoTokenizer fails on this repository
    NAME,
    bos_token="<s>",
    eos_token="</s>",
    sep_token="</s>",
    cls_token="<s>",
    unk_token="<unk>",
    pad_token="<pad>",
    mask_token="<mask>",
)


def split_lead(text: str) -> tuple[str, str] | None:
    """(lead, body) when the lead has 20-150 words and the body at least 150."""
    parts = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(parts) < 2:
        return None
    lead, body = parts[0].strip(), "\n".join(parts[1:]).strip()
    if 20 <= len(lead.split()) <= 150 and len(body.split()) >= 150:
        return lead, body
    return None


class Pairs(torch.utils.data.Dataset):
    def __init__(self, items: list[tuple[str, str]]) -> None:
        self.enc = tok(
            [body for _, body in items],
            text_target=[lead for lead, _ in items],
            truncation=True,
            max_length=512,
        )

    def __len__(self) -> int:
        return len(self.enc["input_ids"])

    def __getitem__(self, index: int) -> dict:
        return {k: v[index] for k, v in self.enc.items()}


def main(n_train: int | None = None) -> None:
    train_pairs = [p for a in load_articles("train") if (p := split_lead(a.text))][:n_train]
    test_pairs = [p for a in load_articles("test") if (p := split_lead(a.text))]
    print(f"{len(train_pairs)} training pairs, {len(test_pairs)} test pairs")
    model = AutoModelForSeq2SeqLM.from_pretrained(NAME)
    args = Seq2SeqTrainingArguments(
        output_dir=str(settings.interim_dir / "ch26_barthez"),
        per_device_train_batch_size=8,
        learning_rate=3e-5,
        num_train_epochs=4,
        weight_decay=0.01,
        report_to="none",
        save_strategy="no",
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=Pairs(train_pairs),
        data_collator=DataCollatorForSeq2Seq(tok, model=model),
    )
    trainer.train()
    lead, body = test_pairs[0]
    enc = tok(body, return_tensors="pt", truncation=True, max_length=512).to(trainer.model.device)
    summary = trainer.model.generate(**enc, num_beams=4, max_new_tokens=80)
    print("Reference:", lead)
    print("BARThez:  ", tok.decode(summary[0], skip_special_tokens=True))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
