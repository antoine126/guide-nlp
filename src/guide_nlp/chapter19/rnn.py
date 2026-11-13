"""A recurrent network from scratch, checked against nn.RNN (chapter 19).

Usage: uv run --group neural python -m guide_nlp.chapter19.rnn
"""

import torch
import torch.nn as nn


class ScratchRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.w_x = nn.Linear(input_size, hidden_size)
        self.w_h = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros(x.size(0), self.w_h.in_features, device=x.device)
        states = []
        for t in range(x.size(1)):
            h = torch.tanh(self.w_h(h) + self.w_x(x[:, t]))
            states.append(h)
        return torch.stack(states, dim=1), h


def main() -> None:
    reference = nn.RNN(8, 16, batch_first=True)
    mine = ScratchRNN(8, 16)
    with torch.no_grad():
        p = dict(reference.named_parameters())  # weight_ih_l0, bias_hh_l0...
        mine.w_x.weight.copy_(p["weight_ih_l0"])
        mine.w_x.bias.copy_(p["bias_ih_l0"] + p["bias_hh_l0"])
        mine.w_h.weight.copy_(p["weight_hh_l0"])
    x = torch.randn(2, 5, 8)
    print(torch.allclose(mine(x)[0], reference(x)[0], atol=1e-6))  # True


if __name__ == "__main__":
    main()
