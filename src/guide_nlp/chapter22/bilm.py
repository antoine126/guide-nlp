"""A bidirectional language model in the style of ELMo (chapter 22).

Usage: uv run --group neural python -m guide_nlp.chapter22.bilm
The training on the corpus is in guide_nlp.chapter22.experiments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiLM(nn.Module):
    """Two independent stacks of LSTMs: left-to-right and right-to-left."""

    def __init__(self, vocab: int, dim: int = 256, hidden: int = 512, layers: int = 2) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, dim, padding_idx=0)
        sizes = [dim] + [hidden] * (layers - 1)
        self.fwd = nn.ModuleList(nn.LSTM(s, hidden, batch_first=True) for s in sizes)
        self.bwd = nn.ModuleList(nn.LSTM(s, hidden, batch_first=True) for s in sizes)
        self.out = nn.Linear(hidden, vocab)
        self.drop = nn.Dropout(0.2)

    def run(self, stack: nn.ModuleList, x: torch.Tensor) -> list[torch.Tensor]:
        states = []
        for lstm in stack:
            x, _ = lstm(self.drop(x))
            states.append(x)
        return states

    def forward(
        self, ids: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        e = self.embed(ids)
        forward_states = self.run(self.fwd, e)
        backward_states = [s.flip(1) for s in self.run(self.bwd, e.flip(1))]
        return e, forward_states, backward_states


def bilm_loss(model: BiLM, ids: torch.Tensor) -> torch.Tensor:
    vocab = model.out.out_features
    _, f, b = model(ids)
    logits_f = model.out(f[-1][:, :-1]).reshape(-1, vocab)  # predicts the next token
    logits_b = model.out(b[-1][:, 1:]).reshape(-1, vocab)  # predicts the previous one
    loss_f = F.cross_entropy(logits_f, ids[:, 1:].reshape(-1), ignore_index=0)
    loss_b = F.cross_entropy(logits_b, ids[:, :-1].reshape(-1), ignore_index=0)
    return loss_f + loss_b


@torch.no_grad()
def contextual_vector(model: BiLM, ids: torch.Tensor, position: int, layer: int) -> torch.Tensor:
    """Vector of the token at `position` (ids has shape (1, T)); layer is 0 or 1."""
    model.eval()
    _, f, b = model(ids)
    return torch.cat([f[layer][0, position], b[layer][0, position]])


def main() -> None:
    torch.manual_seed(0)
    model = BiLM(vocab=1000)
    ids = torch.randint(1, 1000, (4, 12))
    print(
        f"loss of an untrained model: {bilm_loss(model, ids).item():.2f} "
        f"(2 ln 1000 = {2 * torch.log(torch.tensor(1000.0)).item():.2f})"
    )
    print(contextual_vector(model, ids[:1], position=3, layer=1).shape)  # (1024,)


if __name__ == "__main__":
    main()
