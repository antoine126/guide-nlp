"""The short examples of chapter 20: max pooling over time, frWac initialisation, weight tying.

Usage: uv run --group neural python -m guide_nlp.chapter20.snippets
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from guide_nlp.chapter17.corpus import FRWAC, load_frwac


class PooledBiLSTM(nn.Module):
    """A BiLSTM classifier whose document vector is the max over time."""

    def __init__(self, vocab_size: int, dim: int = 128, hidden: int = 128, n: int = 6) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.rnn = nn.LSTM(dim, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.3)
        self.head = nn.Linear(2 * hidden, n)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(
            self.embed(x), lengths, batch_first=True, enforce_sorted=False
        )
        out, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)
        mask = (x != 0).unsqueeze(-1)[:, : out.size(1)]
        pooled = out.masked_fill(~mask, -1e9).max(dim=1).values  # max over time
        return self.head(self.drop(pooled))


def init_from_frwac(embedding: nn.Embedding, stoi: dict[str, int]) -> int:
    """Copy the frWac vector of every known word; returns how many were found."""
    vectors = load_frwac()
    weights = np.random.default_rng(0).normal(0, 0.1, (len(stoi), vectors.vector_size))
    weights[0] = 0  # padding
    found = 0
    for word, i in stoi.items():
        if i > 1 and word in vectors:
            weights[i] = vectors[word]
            found += 1
    embedding.weight.data.copy_(torch.as_tensor(weights, dtype=torch.float32))
    return found


class TiedLM(nn.Module):
    """Weight tying: the output layer reuses the embedding matrix."""

    def __init__(self, vocab_size: int, hidden: int = 256) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden)
        self.rnn = nn.LSTM(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, vocab_size, bias=False)
        self.out.weight = self.embedding.weight  # requires hidden == embedding dimension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, _ = self.rnn(self.embedding(x))
        return self.out(z)


def main() -> None:
    model = PooledBiLSTM(100)
    x = torch.tensor([[5, 6, 7, 0], [8, 9, 0, 0]])
    print(model(x, torch.tensor([3, 2])).shape)  # torch.Size([2, 6])
    tied = TiedLM(1000)
    print(sum(p.numel() for p in tied.parameters()))  # the matrix is counted once
    if FRWAC.exists():
        stoi = {"<pad>": 0, "<unk>": 1, "planète": 2, "zzzz": 3}
        embedding = nn.Embedding(len(stoi), 500, padding_idx=0)
        print(init_from_frwac(embedding, stoi), "of 2 words found in frWac")


if __name__ == "__main__":
    main()
