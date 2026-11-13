"""A neural bag of words: EmbeddingBag + linear layer (chapter 16).

Usage: uv run --group neural python -m guide_nlp.chapter16.bag_classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from guide_nlp.chapter14.data import sentences
from guide_nlp.chapter16.data import PAD, TextDataset
from guide_nlp.chapter16.vocab import Vocab
from guide_nlp.config import settings
from guide_nlp.corpus.load import RUBRICS, WikiArticle, load_articles, words
from guide_nlp.runtime import resolve_device

device = resolve_device(settings.device)


class BagClassifier(nn.Module):
    def __init__(self, vocab_size: int, dim: int, n_classes: int) -> None:
        super().__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, dim, mode="mean", padding_idx=PAD)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(dim, n_classes))

    def forward(self, tokens: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return self.head(self.embedding(tokens, offsets))


def collate_bag(
    batch: list[tuple[torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    documents, labels = zip(*batch, strict=True)
    offsets = torch.tensor([0] + [len(d) for d in documents[:-1]]).cumsum(0)
    return torch.cat(documents), offsets, torch.tensor(labels)


def make_dataset(articles: list[WikiArticle], vocab: Vocab) -> TextDataset:
    documents = [vocab.encode(words(a.text))[:4000] for a in articles]
    return TextDataset(documents, [RUBRICS.index(a.rubric) for a in articles])


@torch.no_grad()
def evaluate(model: BagClassifier, loader: DataLoader) -> tuple[float, float]:
    model.eval()
    predicted, gold = [], []
    for tokens, offsets, labels in loader:
        predicted += model(tokens.to(device), offsets.to(device)).argmax(1).cpu().tolist()
        gold += labels.tolist()
    return accuracy_score(gold, predicted), f1_score(gold, predicted, average="macro")


def main() -> None:
    train, test = load_articles("train"), load_articles("test")
    vocab = Vocab.build(sentences(train))
    train_loader = DataLoader(
        make_dataset(train, vocab), batch_size=32, shuffle=True, collate_fn=collate_bag
    )
    test_loader = DataLoader(make_dataset(test, vocab), batch_size=256, collate_fn=collate_bag)
    for dim in (64, 256):
        torch.manual_seed(0)
        model = BagClassifier(len(vocab), dim, len(RUBRICS)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
        for epoch in range(1, 31):
            model.train()
            for tokens, offsets, labels in train_loader:
                logits = model(tokens.to(device), offsets.to(device))
                loss = F.cross_entropy(logits, labels.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if epoch in (1, 10, 30):
                accuracy, macro_f1 = evaluate(model, test_loader)
                print(f"dim={dim} epoch {epoch}: accuracy={accuracy:.3f} macro F1={macro_f1:.3f}")
    # Book: dim=64 -> 0.925 / 0.929 after 30 epochs; dim=256 -> 0.937 / 0.940


if __name__ == "__main__":
    main()
