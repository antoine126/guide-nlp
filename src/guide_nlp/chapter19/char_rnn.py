"""A character-level language model: RNN or LSTM, and sampling (chapters 19 and 20).

Usage: uv run --group neural python -m guide_nlp.chapter19.char_rnn [RNN|LSTM] [STEPS]
4,000 steps take about 6 minutes (RNN) or 13 minutes (LSTM) on a GPU.
"""

import sys
from collections import Counter
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from guide_nlp.config import settings
from guide_nlp.corpus.load import load_articles
from guide_nlp.runtime import resolve_device

device = resolve_device(settings.device)
text = "\n".join(a.text for a in load_articles("train"))[:3_000_000].lower()
chars = sorted({c for c, n in Counter(text).items() if n >= 50})
char_to_id = {c: i for i, c in enumerate(chars)}


class CharRNN(nn.Module):
    def __init__(self, cell: Callable[..., nn.RNNBase], vocab: int, hidden: int = 512) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, 64)
        self.rnn = cell(64, hidden, batch_first=True, num_layers=2, dropout=0.1)
        self.out = nn.Linear(hidden, vocab)

    def forward(self, x: torch.Tensor, state: Any = None) -> tuple[torch.Tensor, Any]:
        z, state = self.rnn(self.embed(x), state)
        return self.out(z), state


def sample(model: CharRNN, prompt: str, n: int = 200, temperature: float = 0.7) -> str:
    model.eval()
    x = torch.tensor([[char_to_id[c] for c in prompt]], device=device)
    out = prompt
    with torch.no_grad():
        logits, state = model(x)
        for _ in range(n):
            probs = F.softmax(logits[0, -1] / temperature, dim=-1)
            next_id = torch.multinomial(probs, 1)
            out += chars[int(next_id.item())]
            logits, state = model(next_id.view(1, 1), state)
    return out


def main(cell_name: str = "LSTM", n_steps: int = 4000) -> None:
    data = torch.tensor([char_to_id[c] for c in text if c in char_to_id], device=device)
    cell = {"RNN": nn.RNN, "LSTM": nn.LSTM, "GRU": nn.GRU}[cell_name]
    torch.manual_seed(0)
    model = CharRNN(cell, len(chars)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
    seq, batch = 128, 64
    for step in range(1, n_steps + 1):
        model.train()
        starts = torch.randint(0, len(data) - seq - 1, (batch,), device=device)
        x = torch.stack([data[p : p + seq] for p in starts])
        y = torch.stack([data[p + 1 : p + seq + 1] for p in starts])
        logits, _ = model(x)
        loss = F.cross_entropy(logits.reshape(-1, len(chars)), y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step in (500, 2000, n_steps):
            print(f"step {step}: loss={loss.item():.3f} | {sample(model, 'la planète ')!r}")


if __name__ == "__main__":
    main(
        sys.argv[1] if len(sys.argv) > 1 else "LSTM",
        int(sys.argv[2]) if len(sys.argv) > 2 else 4000,
    )
