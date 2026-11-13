"""The neural language model of Bengio et al. (2003) (chapter 16).

Usage: uv run --group neural python -m guide_nlp.chapter16.neural_lm [EPOCHS]
About one minute per epoch on a GPU; the test perplexity is lowest after
epoch 3 (229.4 in the book), then the model overfits.
"""

import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from guide_nlp.chapter14.data import sentences
from guide_nlp.chapter16.vocab import BOS, EOS, UNK, Vocab
from guide_nlp.config import settings
from guide_nlp.corpus.load import WikiArticle, load_articles
from guide_nlp.runtime import resolve_device

device = resolve_device(settings.device)
CONTEXT = 3


class NeuralLM(nn.Module):
    """Bengio et al. (2003): concatenated context embeddings + MLP."""

    def __init__(
        self,
        vocab_size: int,
        context: int = 3,
        dim: int = 64,
        hidden: int = 512,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.mlp = nn.Sequential(
            nn.Linear(context * dim, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, vocab_size),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.embedding(context).flatten(1))


@torch.no_grad()
def perplexity(model: nn.Module, X: torch.Tensor, Y: torch.Tensor, batch_size: int = 8192) -> float:
    model.eval()
    total = 0.0
    for i in range(0, len(X), batch_size):
        logits = model(X[i : i + batch_size].to(device))
        targets = Y[i : i + batch_size].to(device)
        total += F.cross_entropy(logits, targets, reduction="sum").item()
    return math.exp(total / len(X))


def windows(articles: list[WikiArticle], vocab: Vocab) -> tuple[torch.Tensor, torch.Tensor]:
    """Every (3 previous words, next word) pair, sentence by sentence."""
    xs, ys = [], []
    for s in sentences(articles):
        q = [BOS] * CONTEXT + vocab.encode(s) + [EOS]
        for i in range(CONTEXT, len(q)):
            xs.append(q[i - CONTEXT : i])
            ys.append(q[i])
    return torch.tensor(xs), torch.tensor(ys)


@torch.no_grad()
def complete(
    model: NeuralLM, vocab: Vocab, prompt: str, n: int = 20, temperature: float = 0.8
) -> str:
    """Sample a continuation, word by word."""
    model.eval()
    context = ([BOS] * CONTEXT + vocab.encode(prompt.split()))[-CONTEXT:]
    out = prompt.split()
    for _ in range(n):
        p = F.softmax(model(torch.tensor([context], device=device))[0] / temperature, -1)
        p[UNK] = 0
        i = int(torch.multinomial(p, 1).item())
        if i == EOS:
            break
        out.append(vocab.itos[i])
        context = [*context[1:], i]
    return " ".join(out)


def main(n_epochs: int = 8) -> None:
    torch.manual_seed(0)
    train, test = load_articles("train"), load_articles("test")
    vocab = Vocab.build(sentences(train))
    X_train, Y_train = windows(train, vocab)
    X_test, Y_test = windows(test, vocab)
    print(f"vocabulary {len(vocab)}, {len(X_train)} training windows")

    model = NeuralLM(len(vocab)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(n_epochs):
        model.train()
        order = torch.randperm(len(X_train))
        for i in range(0, len(X_train), 1024):
            batch = order[i : i + 1024]
            logits = model(X_train[batch].to(device))
            loss = F.cross_entropy(logits, Y_train[batch].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(epoch, f"{perplexity(model, X_test, Y_test):.1f}")

    torch.manual_seed(0)
    for prompt in ("la planète", "le président de", "le virus"):
        print("sample:", complete(model, vocab, prompt))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 8)
