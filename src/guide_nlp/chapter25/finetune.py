"""Fine-tuning CamemBERT to classify rubrics with the Trainer (chapter 25).

Usage: uv run --group neural python -m guide_nlp.chapter25.finetune [N_TRAIN]
Three epochs on the 2,339 training articles take a few minutes on a GPU
(macro F1 0.961 in the book).
"""

import sys

import torch
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from guide_nlp.config import settings
from guide_nlp.corpus.load import RUBRICS, load_articles

NAME = "almanach/camembert-base"
tokenizer = AutoTokenizer.from_pretrained(NAME)


class Encoded(torch.utils.data.Dataset):
    def __init__(self, texts: list[str], targets: list[int]) -> None:
        self.encodings = tokenizer(texts, truncation=True, max_length=256)
        self.targets = targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> dict:
        return {**{k: v[index] for k, v in self.encodings.items()}, "labels": self.targets[index]}


def main(n_train: int | None = None) -> None:
    train, test = load_articles("train")[:n_train], load_articles("test")
    train_texts, train_labels = [a.text for a in train], [RUBRICS.index(a.rubric) for a in train]
    test_texts, test_labels = [a.text for a in test], [RUBRICS.index(a.rubric) for a in test]

    torch.manual_seed(0)
    model = AutoModelForSequenceClassification.from_pretrained(NAME, num_labels=6)
    args = TrainingArguments(
        output_dir=str(settings.interim_dir / "ch25_finetune"),
        per_device_train_batch_size=16,
        learning_rate=2e-5,
        num_train_epochs=3,
        warmup_steps=44,
        weight_decay=0.01,
        report_to="none",
        save_strategy="no",
        seed=0,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=Encoded(train_texts, train_labels),
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    trainer.train()
    predictions = trainer.predict(Encoded(test_texts, test_labels)).predictions.argmax(-1)
    print(f"macro F1 = {f1_score(test_labels, predictions, average='macro'):.3f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
