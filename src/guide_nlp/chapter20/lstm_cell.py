"""An LSTM cell from scratch, checked against nn.LSTM; the forget-gate bias (chapter 20).

Usage: uv run --group neural python -m guide_nlp.chapter20.lstm_cell
"""

import torch
import torch.nn as nn


class ScratchLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(
        self, x: torch.Tensor, state: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, c = state
        i, f, g, o = self.linear(torch.cat([x, h], dim=-1)).chunk(4, dim=-1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


def set_forget_bias(lstm: nn.LSTM, value: float) -> None:
    """PyTorch orders the gates (input, forget, cell, output) in each bias vector."""
    hidden = lstm.hidden_size
    with torch.no_grad():
        for name, param in lstm.named_parameters():
            if "bias" in name:
                param[hidden : 2 * hidden] = value / 2  # bias_ih + bias_hh add up


def main() -> None:
    reference = nn.LSTM(8, 16, batch_first=True)
    cell = ScratchLSTMCell(8, 16)
    with torch.no_grad():
        p = dict(reference.named_parameters())  # weight_ih_l0, bias_hh_l0...
        weights = torch.cat([p["weight_ih_l0"], p["weight_hh_l0"]], dim=1)
        cell.linear.weight.copy_(weights)
        cell.linear.bias.copy_(p["bias_ih_l0"] + p["bias_hh_l0"])

    x = torch.randn(2, 5, 8)
    h = c = torch.zeros(2, 16)
    outputs = []
    for t in range(5):
        h, c = cell(x[:, t], (h, c))
        outputs.append(h)
    print(torch.allclose(torch.stack(outputs, 1), reference(x)[0], atol=1e-6))  # True

    for name, cls in (("RNN", nn.RNN), ("GRU", nn.GRU), ("LSTM", nn.LSTM)):
        print(name, sum(p.numel() for p in cls(64, 128).parameters()))
    # RNN 24832, GRU 74496, LSTM 99328: one, three and four times h(h + d) + 2h


if __name__ == "__main__":
    main()
