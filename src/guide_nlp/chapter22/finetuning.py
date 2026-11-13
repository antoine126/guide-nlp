"""ULMFiT's tricks: gradual unfreezing and discriminative learning rates (chapter 22).

Usage: uv run --group neural python -m guide_nlp.chapter22.finetuning
"""

import torch
import torch.nn as nn


def freeze_all_but(model: nn.Module, trainable: list[nn.Module]) -> None:
    for p in model.parameters():
        p.requires_grad = False
    for module in trainable:
        for p in module.parameters():
            p.requires_grad = True


def layer_groups_lr(groups: list[nn.Module], lr: float, factor: float = 2.6) -> list[dict]:
    """Discriminative rates: the last group gets lr, each lower group lr / factor."""
    n = len(groups)
    return [
        {"params": g.parameters(), "lr": lr / factor ** (n - 1 - i)} for i, g in enumerate(groups)
    ]


def main() -> None:
    groups: list[nn.Module] = [nn.Embedding(100, 16), nn.LSTM(16, 32), nn.Linear(32, 6)]
    model = nn.ModuleList(groups)
    freeze_all_but(model, [groups[-1]])  # step 1: train the head only
    print([any(p.requires_grad for p in g.parameters()) for g in groups])
    optimizer = torch.optim.AdamW(layer_groups_lr(groups, lr=1e-3))
    print([f"{g['lr']:.2e}" for g in optimizer.param_groups])


if __name__ == "__main__":
    main()
