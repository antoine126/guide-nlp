"""The short examples of chapter 16, in the order of the text.

Usage: uv run --group neural python -m guide_nlp.chapter16.snippets
"""

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


def softmax_gradient() -> None:
    """The gradient of softmax + cross-entropy is p - y."""
    logits = torch.tensor([1.0, 2.0, 0.5], requires_grad=True)
    loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([1]))
    loss.backward()
    assert logits.grad is not None
    print(round(loss.item(), 4))  # 0.4644
    print([round(g, 4) for g in logits.grad.tolist()])  # [0.2312, -0.3715, 0.1402]
    print((F.softmax(logits, 0) - torch.tensor([0.0, 1.0, 0.0])).tolist())  # the same values


def embedding_is_a_lookup() -> None:
    """An embedding layer is a one-hot vector times a matrix, without the product."""
    E = nn.Embedding(5, 3)
    x = torch.tensor([2])
    one_hot = F.one_hot(x, 5).float()
    print(torch.allclose(E(x), one_hot @ E.weight))  # True


def train_epochs(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
    device: torch.device,
    evaluate: Callable[[nn.Module], None],
) -> None:
    """The canonical PyTorch training loop."""
    for _ in range(n_epochs):
        model.train()  # enable dropout
        for tokens, labels in loader:
            loss = F.cross_entropy(model(tokens.to(device)), labels.to(device))
            optimizer.zero_grad()  # otherwise the gradients accumulate
            loss.backward()
            optimizer.step()
        model.eval()  # disable dropout before evaluating
        evaluate(model)


def main() -> None:
    for demo in (softmax_gradient, embedding_is_a_lookup):
        print(f"--- {demo.__name__}")
        demo()


if __name__ == "__main__":
    main()
