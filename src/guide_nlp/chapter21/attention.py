"""Additive attention (Bahdanau et al., 2015) (chapter 21)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    """Bahdanau et al. (2015): e = v^T tanh(W s + U h)."""

    def __init__(self, dec_size: int, enc_size: int, attn_size: int = 256) -> None:
        super().__init__()
        self.W = nn.Linear(dec_size, attn_size, bias=False)
        self.U = nn.Linear(enc_size, attn_size, bias=False)
        self.v = nn.Linear(attn_size, 1, bias=False)

    def forward(
        self, s: torch.Tensor, enc_out: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.tanh(self.W(s).unsqueeze(1) + self.U(enc_out))
        scores = self.v(hidden).squeeze(-1)
        alpha = F.softmax(scores.masked_fill(~mask, -1e9), dim=-1)
        context = (alpha.unsqueeze(1) @ enc_out).squeeze(1)
        return context, alpha
