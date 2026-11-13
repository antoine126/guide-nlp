"""A Transformer encoder from scratch, trained to classify rubrics (chapter 24).

Usage: uv run --group neural python -m guide_nlp.chapter24.mini_encoder [EPOCHS]
"""

import math
import sys
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from guide_nlp.config import settings
from guide_nlp.corpus.load import RUBRICS, load_articles, words
from guide_nlp.runtime import resolve_device

device = resolve_device(settings.device)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.d_k = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        # (3, batch, heads, n, d_k)
        q, k, v = self.qkv(x).view(b, n, 3, self.n_heads, self.d_k).permute(2, 0, 3, 1, 4)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        scores = scores.masked_fill(padding_mask[:, None, None, :], float("-inf"))
        z = F.softmax(scores, dim=-1) @ v
        return self.out(z.transpose(1, 2).reshape(b, n, -1))


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.attn(self.norm1(x), mask))  # pre-LN
        return x + self.drop(self.ff(self.norm2(x)))


class MiniEncoder(nn.Module):
    def __init__(
        self,
        vocab: int,
        n_classes: int,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 256,
        layers: int = 2,
        dropout: float = 0.2,
        max_len: int = 512,
    ) -> None:
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model, padding_idx=0)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(
            EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(layers)
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        mask = ids == 0  # True on [PAD]
        x = self.tok(ids) + self.pos(torch.arange(ids.size(1), device=ids.device))
        for block in self.blocks:
            x = block(x, mask)
        x = self.norm(x).masked_fill(mask[..., None], 0)
        return self.head(x.sum(1) / (~mask).sum(1, keepdim=True))  # mean pooling


def encode(texts: list[str], stoi: dict[str, int], max_len: int = 256) -> torch.Tensor:
    ids = torch.zeros(len(texts), max_len, dtype=torch.long)
    for i, text in enumerate(texts):
        tokens = [stoi.get(w, 1) for w in words(text)[:max_len]] or [1]
        ids[i, : len(tokens)] = torch.tensor(tokens)
    return ids


def main(n_epochs: int = 20) -> None:
    train, test = load_articles("train"), load_articles("test")
    counts = Counter(w for a in train for w in words(a.text))
    stoi = {w: i + 2 for i, (w, c) in enumerate(counts.most_common()) if c >= 3}
    X_train, X_test = encode([a.text for a in train], stoi), encode([a.text for a in test], stoi)
    y_train = torch.tensor([RUBRICS.index(a.rubric) for a in train])
    y_test = [RUBRICS.index(a.rubric) for a in test]
    torch.manual_seed(0)
    model = MiniEncoder(len(stoi) + 2, len(RUBRICS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    for epoch in range(1, n_epochs + 1):
        model.train()
        order = torch.randperm(len(X_train))
        for i in range(0, len(order), 32):
            batch = order[i : i + 32]
            loss = F.cross_entropy(model(X_train[batch].to(device)), y_train[batch].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                predicted = model(X_test.to(device)).argmax(1).cpu().tolist()
            print(f"epoch {epoch}: macro F1={f1_score(y_test, predicted, average='macro'):.3f}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
