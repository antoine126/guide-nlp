"""An encoder-decoder with attention for French -> English (chapter 21).

Usage: uv run --group neural python -m guide_nlp.chapter21.seq2seq
The full training on Tatoeba is in guide_nlp.chapter21.experiments.
"""

import re

import torch
import torch.nn as nn

from guide_nlp.chapter21.attention import AdditiveAttention

PAD, BOS, EOS, UNK = 0, 1, 2, 3
TOKEN = re.compile(r"[^\W\d_]+'?|\d+|[^\w\s]")


def tokenize(sentence: str) -> list[str]:
    """Lowercased words, elisions kept with their apostrophe, punctuation apart."""
    sentence = sentence.lower().replace("’", "'").replace("\xa0", " ").replace(" ", " ")
    return TOKEN.findall(sentence)


class Encoder(nn.Module):
    def __init__(self, vocab: int, embed: int = 128, hidden: int = 256) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, embed, padding_idx=PAD)
        self.rnn = nn.GRU(embed, hidden, batch_first=True, bidirectional=True)
        self.bridge = nn.Linear(2 * hidden, hidden)

    def forward(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lengths = (src != PAD).sum(1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            self.embed(src), lengths, batch_first=True, enforce_sorted=False
        )
        out, h = self.rnn(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        return out, torch.tanh(self.bridge(torch.cat([h[0], h[1]], dim=-1)))


class Seq2Seq(nn.Module):
    def __init__(self, src_vocab: int, tgt_vocab: int, embed: int = 128, hidden: int = 256) -> None:
        super().__init__()
        self.encoder = Encoder(src_vocab, embed, hidden)
        self.attention = AdditiveAttention(hidden, 2 * hidden)
        self.embed = nn.Embedding(tgt_vocab, embed, padding_idx=PAD)
        self.cell = nn.GRUCell(embed + 2 * hidden, hidden)
        self.out = nn.Linear(hidden + 2 * hidden + embed, tgt_vocab)

    def step(
        self, token: torch.Tensor, s: torch.Tensor, enc_out: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        e = self.embed(token)
        context, alpha = self.attention(s, enc_out, mask)
        s = self.cell(torch.cat([e, context], -1), s)
        return self.out(torch.cat([s, context, e], -1)), s, alpha

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        enc_out, s = self.encoder(src)
        mask = src != PAD
        logits = []
        for t in range(tgt.size(1) - 1):  # teacher forcing: tgt[:, t] is true
            step_logits, s, _ = self.step(tgt[:, t], s, enc_out, mask)
            logits.append(step_logits)
        return torch.stack(logits, 1)


def main() -> None:
    print(tokenize("Il n’a pas acheté la voiture rouge."))
    model = Seq2Seq(src_vocab=100, tgt_vocab=120)
    src = torch.tensor([[5, 6, 7, 0], [8, 9, 10, 11]])
    tgt = torch.tensor([[BOS, 4, 5, EOS], [BOS, 6, EOS, PAD]])
    print(model(src, tgt).shape)  # (2, 3, 120): one distribution per target step
    print(sum(p.numel() for p in model.parameters()), "parameters")


if __name__ == "__main__":
    main()
