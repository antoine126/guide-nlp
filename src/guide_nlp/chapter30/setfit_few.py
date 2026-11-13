"""SetFit: a classifier from eight examples per rubric (chapter 30).

Usage: uv run --group neural python -m guide_nlp.chapter30.setfit_few [N_PER_RUBRIC]
"""

import random
import sys

import numpy as np
from datasets import Dataset
from setfit import SetFitModel, Trainer, TrainingArguments
from sklearn.metrics import f1_score

from guide_nlp.corpus.load import RUBRICS, load_articles


def main(per_rubric: int = 8) -> None:
    random.seed(0)
    train = load_articles("train")
    examples = [
        a for r in RUBRICS for a in random.sample([a for a in train if a.rubric == r], per_rubric)
    ]
    texts = [" ".join(a.text.split()[:200]) for a in examples]
    labels = [RUBRICS.index(a.rubric) for a in examples]

    model = SetFitModel.from_pretrained(
        "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    )
    args = TrainingArguments(batch_size=16, num_epochs=1, num_iterations=20)
    trainer = Trainer(
        model=model, args=args, train_dataset=Dataset.from_dict({"text": texts, "label": labels})
    )
    trainer.train()
    print(model.predict(["Le télescope a observé une nouvelle galaxie."]))
    test = load_articles("test")
    predicted = model.predict([" ".join(a.text.split()[:200]) for a in test])
    gold = [RUBRICS.index(a.rubric) for a in test]
    print(
        f"{per_rubric} examples per rubric: macro F1 = "
        f"{f1_score(gold, np.asarray(predicted).tolist(), average='macro'):.3f}"
    )


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
