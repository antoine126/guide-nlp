"""Token classification: aligning word labels with sub-tokens (chapter 25).

Usage: uv run --group neural python -m guide_nlp.chapter25.ner [N_TRAIN]
Fine-tunes CamemBERT on WikiNER, then tags two sentences.
"""

import sys
from typing import Any

import torch
from datasets import load_dataset
from seqeval.metrics import f1_score
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from guide_nlp.config import settings

NAME = "almanach/camembert-base"
tok = AutoTokenizer.from_pretrained(NAME)


def encode_aligned(batch: dict[str, Any]) -> Any:
    """Label the first sub-token of each word; -100 elsewhere (ignored by the loss)."""
    enc = tok(batch["tokens"], is_split_into_words=True, truncation=True, max_length=256)
    all_labels = []
    for i, tags in enumerate(batch["ner_tags"]):
        previous, row = None, []
        for word in enc.word_ids(batch_index=i):
            row.append(-100 if word is None or word == previous else tags[word])
            previous = word
        all_labels.append(row)
    enc["labels"] = all_labels
    return enc


@torch.no_grad()
def tag(
    model: AutoModelForTokenClassification, words: list[str], names: list[str]
) -> list[tuple[str, str]]:
    enc = tok(words, is_split_into_words=True, return_tensors="pt").to(model.device)
    predicted = model(**enc).logits[0].argmax(-1).tolist()
    out, previous = [], None
    for position, word in enumerate(enc.word_ids()):
        if word is not None and word != previous:
            out.append((words[word], names[predicted[position]]))
        previous = word
    return out


def main(n_train: int = 20_000) -> None:
    data = load_dataset("Jean-Baptiste/wikiner_fr")
    names = data["train"].features["ner_tags"].feature.names  # O, LOC, PER, MISC, ORG
    train = data["train"].shuffle(seed=0).select(range(n_train))
    test = data["test"].select(range(2000))
    columns = train.column_names
    train_enc = train.map(encode_aligned, batched=True, remove_columns=columns)
    test_enc = test.map(encode_aligned, batched=True, remove_columns=columns)

    torch.manual_seed(0)
    model = AutoModelForTokenClassification.from_pretrained(NAME, num_labels=len(names))
    args = TrainingArguments(
        output_dir=str(settings.interim_dir / "ch25_ner"),
        per_device_train_batch_size=32,
        learning_rate=3e-5,
        num_train_epochs=1,
        warmup_steps=max(n_train // 32 // 10, 1),
        weight_decay=0.01,
        report_to="none",
        save_strategy="no",
        seed=0,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_enc,
        data_collator=DataCollatorForTokenClassification(tok),
    )
    trainer.train()

    output = trainer.predict(test_enc)
    gold, predicted = [], []
    for row_pred, row_gold in zip(output.predictions.argmax(-1), output.label_ids, strict=True):
        pairs = [(names[p], names[g]) for p, g in zip(row_pred, row_gold, strict=True) if g != -100]
        # WikiNER uses IO tags: seqeval reads them as I- entities.
        predicted.append([f"I-{p}" if p != "O" else p for p, _ in pairs])
        gold.append([f"I-{g}" if g != "O" else g for _, g in pairs])
    print(f"entity F1 = {f1_score(gold, predicted):.3f}")
    for sentence in (
        "La société Acme a racheté la start-up lyonnaise Datalab en 2023 .",
        "Emmanuel Macron a inauguré mardi le nouveau laboratoire du CNRS à Grenoble .",
    ):
        print([(w, t) for w, t in tag(trainer.model, sentence.split(), names) if t != "O"])


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20_000)
