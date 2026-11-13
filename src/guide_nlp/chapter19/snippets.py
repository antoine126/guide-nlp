"""The short examples of chapter 19: gradient profile, shapes, packing.

Usage: uv run --group neural python -m guide_nlp.chapter19.snippets
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


def gradient_profile(net: nn.RNNBase, length: int = 100) -> torch.Tensor:
    """Norm of d(last output)/d(input at t), indexed by the distance to the last step."""
    x = torch.randn(1, length, net.input_size, requires_grad=True)
    output, _ = net(x)
    output[0, -1].sum().backward()
    assert x.grad is not None
    return x.grad[0].norm(dim=-1).flip(0)  # index = distance to the last step


def shapes() -> None:
    rnn = nn.RNN(input_size=64, hidden_size=128, num_layers=2, bidirectional=True, batch_first=True)
    x = torch.randn(32, 50, 64)  # (batch, time, features)
    output, h_n = rnn(x)
    print(output.shape)  # (32, 50, 256): last layer, every step, both directions
    print(h_n.shape)  # (4, 32, 128): last step, every layer and direction


def packing() -> None:
    """Padding pollutes the last hidden state; packing does not."""
    torch.manual_seed(0)
    rnn = nn.GRU(4, 8, batch_first=True)
    sequence = torch.randn(1, 3, 4)
    padded = torch.cat([sequence, torch.zeros(1, 7, 4)], dim=1)
    _, h_true = rnn(sequence)
    _, h_padded = rnn(padded)
    lengths = torch.tensor([3])
    packed = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)
    _, h_packed = rnn(packed)
    print(f"{(h_padded - h_true).norm():.4f}")  # nonzero (0.4947 with the book's seed)
    print(f"{(h_packed - h_true).norm():.4f}")  # 0.0000


def main() -> None:
    torch.manual_seed(0)
    profile = gradient_profile(nn.RNN(16, 64, batch_first=True))
    print([f"{profile[lag].item():.1e}" for lag in (0, 10, 50, 99)])
    shapes()
    packing()


if __name__ == "__main__":
    main()
