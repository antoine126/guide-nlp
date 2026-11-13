"""GloVe from scratch: weighted least squares on log co-occurrences (chapter 18).

Usage: uv run --group neural python -m guide_nlp.chapter18.glove
"""

from collections import defaultdict

import numpy as np
import torch

from guide_nlp.chapter17.corpus import train_sentences, word_counts
from guide_nlp.config import settings
from guide_nlp.runtime import resolve_device

device = resolve_device(settings.device)
PROBES = ["planète", "galaxie", "roi", "virus", "banque", "football", "ordinateur", "lundi"]


def cooccurrences(
    sentences: list[list[str]], stoi: dict[str, int], window: int = 10
) -> tuple[list[int], list[int], list[float]]:
    """Symmetric counts weighted by 1/distance, as in the GloVe paper."""
    table: defaultdict[tuple[int, int], float] = defaultdict(float)
    for s in sentences:
        ids = [stoi.get(t, -1) for t in s]
        for i, a in enumerate(ids):
            if a < 0:
                continue
            for j in range(i + 1, min(i + window + 1, len(ids))):
                b = ids[j]
                if b >= 0:
                    table[(a, b)] += 1.0 / (j - i)
                    table[(b, a)] += 1.0 / (j - i)
    rows, cols = zip(*table.keys(), strict=True)
    return list(rows), list(cols), list(table.values())


def train_glove(
    rows: list[int], cols: list[int], counts: list[float], V: int, n_epochs: int = 50
) -> torch.Tensor:
    torch.manual_seed(0)
    X = torch.tensor(counts, device=device)  # the nonzero co-occurrences
    I, J = torch.tensor(rows, device=device), torch.tensor(cols, device=device)
    weight = torch.clamp(X / 100.0, max=1.0) ** 0.75  # f(x), x_max = 100
    log_x = torch.log(X)

    W = torch.nn.Parameter(torch.randn(V, 100, device=device) * 0.1)
    C = torch.nn.Parameter(torch.randn(V, 100, device=device) * 0.1)
    b_w = torch.nn.Parameter(torch.zeros(V, device=device))
    b_c = torch.nn.Parameter(torch.zeros(V, device=device))
    optimizer = torch.optim.Adagrad([W, C, b_w, b_c], lr=0.05)

    for epoch in range(n_epochs):
        order = torch.randperm(len(X), device=device)
        total = 0.0
        for k in range(0, len(X), 200_000):
            idx = order[k : k + 200_000]
            prediction = (W[I[idx]] * C[J[idx]]).sum(-1) + b_w[I[idx]] + b_c[J[idx]]
            loss = (weight[idx] * (prediction - log_x[idx]) ** 2).sum() / len(idx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        if epoch + 1 in (1, 10, n_epochs):
            print(f"epoch {epoch + 1}: J/N={total / len(X):.4f}")
    return torch.nn.functional.normalize((W + C).detach(), dim=1)  # sum of both tables


def main() -> None:
    counts = word_counts()
    itos = [w for w, c in counts.most_common() if c >= 5]
    stoi = {w: i for i, w in enumerate(itos)}
    rows, cols, values = cooccurrences(train_sentences(), stoi)
    print(f"vocabulary {len(itos)}, {len(values)} nonzero co-occurrences")
    vectors = train_glove(rows, cols, values, len(itos)).cpu().numpy()
    for word in PROBES:
        nearest = np.argsort(-(vectors @ vectors[stoi[word]]))[1:9]
        print(f"{word:<11} -> {[itos[i] for i in nearest]}")


if __name__ == "__main__":
    main()
