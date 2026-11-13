"""The short examples of chapter 25: the CamemBERTv2 tokenizer, frozen [CLS] features.

Usage: uv run --group neural python -m guide_nlp.chapter25.snippets
"""

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast

from guide_nlp.corpus.load import load_articles


def camembert_v2_tokenizer() -> None:
    """With Transformers 5, AutoTokenizer reads these repositories character by character."""
    tok = AutoTokenizer.from_pretrained("almanach/camembertv2-base")
    print(tok.convert_ids_to_tokens(tok("Le camembert est un fromage.")["input_ids"])[:8])
    # ['[CLS]', 'Ġ', 'L', 'e', 'Ġ', 'c', 'a', 'm', ...]
    # reads tokenizer.json
    tok = PreTrainedTokenizerFast.from_pretrained("almanach/camembertv2-base")
    print(tok.convert_ids_to_tokens(tok("Le camembert est un fromage.")["input_ids"]))
    # ['[CLS]', 'Le', 'cam', '##em', '##bert', 'est', 'un', 'fromage', '.', '[SEP]']


def frozen_features(n_train: int = 600) -> None:
    """The [CLS] vector of a frozen encoder, fed to a logistic regression."""
    tokenizer = AutoTokenizer.from_pretrained("almanach/camembert-base")
    encoder = AutoModel.from_pretrained("almanach/camembert-base").eval()

    @torch.no_grad()
    def features(texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 64):
            enc = tokenizer(
                texts[i : i + 64],
                truncation=True,
                max_length=256,
                padding=True,
                return_tensors="pt",
            )
            out.append(encoder(**enc).last_hidden_state[:, 0])  # the [CLS] vector
        return torch.cat(out).numpy()

    train, test = load_articles("train")[:n_train], load_articles("test")[:200]
    classifier = LogisticRegression(max_iter=3000).fit(
        features([a.text for a in train]), [a.rubric for a in train]
    )
    predicted = classifier.predict(features([a.text for a in test]))
    print(
        f"frozen [CLS] + logistic regression: macro F1 = "
        f"{f1_score([a.rubric for a in test], predicted, average='macro'):.3f}"
    )


def main() -> None:
    camembert_v2_tokenizer()
    frozen_features()


if __name__ == "__main__":
    main()
