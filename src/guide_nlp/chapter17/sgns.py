"""Skip-gram with negative sampling, from scratch in PyTorch (chapter 17).

Usage: uv run --group neural python -m guide_nlp.chapter17.sgns
Five epochs take a few minutes on a GPU.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from guide_nlp.chapter17.corpus import train_sentences, word_counts
from guide_nlp.config import settings
from guide_nlp.runtime import resolve_device

device = resolve_device(settings.device)
PROBES = ["planète", "galaxie", "roi", "virus", "banque", "football", "ordinateur", "guerre"]

counts = word_counts()
MIN_COUNT, WINDOW, K, DIM, T_SUB = 5, 5, 5, 100, 1e-4
itos = [w for w, c in counts.most_common() if c >= MIN_COUNT]
stoi = {w: i for i, w in enumerate(itos)}
freq = np.array([counts[w] for w in itos], dtype=float)
keep_prob = np.minimum(1.0, np.sqrt(T_SUB / (freq / freq.sum())))
noise = torch.tensor(freq**0.75, dtype=torch.float32, device=device)
noise = noise / noise.sum()
rng = np.random.default_rng(0)


def make_pairs(sentences: list[list[str]]) -> tuple[torch.Tensor, torch.Tensor]:
    centers, contexts = [], []
    for sentence in sentences:
        ids = [stoi[t] for t in sentence if t in stoi]
        ids = [i for i in ids if rng.random() < keep_prob[i]]  # subsampling
        for pos, center in enumerate(ids):
            r = rng.integers(1, WINDOW + 1)  # dynamic window
            for other in ids[max(0, pos - r) : pos] + ids[pos + 1 : pos + 1 + r]:
                centers.append(center)
                contexts.append(other)
    return torch.tensor(centers), torch.tensor(contexts)


class SGNS(nn.Module):
    def __init__(self, vocab_size: int, dim: int) -> None:
        super().__init__()
        self.center = nn.Embedding(vocab_size, dim)
        self.context = nn.Embedding(vocab_size, dim)
        nn.init.uniform_(self.center.weight, -0.5 / dim, 0.5 / dim)
        nn.init.zeros_(self.context.weight)

    def forward(self, c: torch.Tensor, o: torch.Tensor, neg: torch.Tensor) -> torch.Tensor:
        u, v = self.center(c), self.context(o)
        positive = F.logsigmoid((u * v).sum(-1))
        scores = (self.context(neg) @ u.unsqueeze(-1)).squeeze(-1)
        negative = F.logsigmoid(-scores).sum(-1)
        return -(positive + negative).mean()


def train(n_epochs: int = 5) -> SGNS:
    sentences = train_sentences()
    torch.manual_seed(0)
    model = SGNS(len(itos), DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    for epoch in range(n_epochs):
        c, o = make_pairs(sentences)  # new subsampling at every epoch
        order = torch.randperm(len(c))
        total = 0.0
        for i in range(0, len(c), 8192):
            batch = order[i : i + 8192]
            neg = torch.multinomial(noise, len(batch) * K, replacement=True)
            neg = neg.view(len(batch), K)
            loss = model(c[batch].to(device), o[batch].to(device), neg)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        print(f"epoch {epoch + 1}: {len(c)} pairs, loss={total / len(c):.3f}")
    return model


def main() -> None:
    print(
        f"vocabulary {len(itos)}; keep probability of de/la/planète:",
        [round(float(keep_prob[stoi[w]]), 3) for w in ("de", "la", "planète")],
    )
    model = train()
    E = F.normalize(model.center.weight.detach(), dim=1).cpu().numpy()
    for word in PROBES:
        print(f"{word:<11} -> {[itos[i] for i in np.argsort(-(E @ E[stoi[word]]))[1:9]]}")


if __name__ == "__main__":
    main()
