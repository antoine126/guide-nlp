"""Recurrent classifiers of rubrics, with packed sequences (chapters 19 and 20).

Usage: uv run --group neural python -m guide_nlp.chapter19.classifier [RNN|GRU|LSTM] [--bi]
"""

import random
import sys
from collections import Counter
from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.nn.utils.rnn import pack_padded_sequence

from guide_nlp.config import settings
from guide_nlp.corpus.load import RUBRICS, WikiArticle, load_articles, words
from guide_nlp.runtime import resolve_device

device = resolve_device(settings.device)
CELLS: dict[str, Callable[..., nn.RNNBase]] = {"RNN": nn.RNN, "GRU": nn.GRU, "LSTM": nn.LSTM}
MAXLEN = 200
Example = tuple[list[int], int]


class RecurrentClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        dim: int = 128,
        hidden: int = 128,
        n: int = 6,
        cell: str = "RNN",
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.rnn = CELLS[cell](dim, hidden, batch_first=True, bidirectional=bidirectional)
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(hidden * (2 if bidirectional else 1), n)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(
            self.embed(x), lengths, batch_first=True, enforce_sorted=False
        )
        _, h = self.rnn(packed)
        if isinstance(h, tuple):  # LSTM: (h, c)
            h = h[0]
        last = torch.cat([h[-2], h[-1]], -1) if self.rnn.bidirectional else h[-1]
        return self.head(self.drop(last))


def encode_corpus() -> tuple[list[str], list[Example], list[Example]]:
    train, test = load_articles("train"), load_articles("test")
    counts = Counter(w for a in train for w in words(a.text))
    itos = ["<pad>", "<unk>"] + [w for w, c in counts.most_common() if c >= 3]
    stoi = {w: i for i, w in enumerate(itos)}

    def encode(a: WikiArticle) -> Example:
        ids = [stoi.get(w, 1) for w in words(a.text)[:MAXLEN]] or [1]
        return ids, RUBRICS.index(a.rubric)

    return itos, [encode(a) for a in train], [encode(a) for a in test]


def collate(batch: list[Example]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([len(x) for x, _ in batch])
    X = torch.zeros(len(batch), int(lengths.max()), dtype=torch.long)
    for i, (x, _) in enumerate(batch):
        X[i, : len(x)] = torch.tensor(x)
    return X.to(device), lengths, torch.tensor([y for _, y in batch], device=device)


@torch.no_grad()
def macro_f1(model: nn.Module, examples: list[Example]) -> float:
    model.eval()
    predicted, gold = [], []
    for i in range(0, len(examples), 128):
        x, lengths, y = collate(examples[i : i + 128])
        predicted += model(x, lengths).argmax(1).tolist()
        gold += y.tolist()
    return float(f1_score(gold, predicted, average="macro"))


def train(
    model: nn.Module, train_set: list[Example], test_set: list[Example], n_epochs: int = 15
) -> list[float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    curve = []
    for _ in range(n_epochs):
        model.train()
        random.shuffle(train_set)
        for i in range(0, len(train_set), 32):
            x, lengths, y = collate(train_set[i : i + 32])
            loss = F.cross_entropy(model(x, lengths), y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        curve.append(round(macro_f1(model, test_set), 3))
    return curve


def main() -> None:
    cell = next((a for a in sys.argv[1:] if a in CELLS), "LSTM")
    bidirectional = "--bi" in sys.argv
    itos, train_set, test_set = encode_corpus()
    torch.manual_seed(0)
    random.seed(0)
    model = RecurrentClassifier(len(itos), cell=cell, bidirectional=bidirectional).to(device)
    curve = train(model, train_set, test_set)
    print(f"{'Bi' if bidirectional else ''}{cell}: macro F1 by epoch {curve}")
    # Book, table 19.2: RNN 0.34, GRU 0.726 (best), LSTM 0.683, BiLSTM 0.750


if __name__ == "__main__":
    main()
