"""Beam search with length normalisation (chapter 21).

Usage: uv run --group neural python -m guide_nlp.chapter21.beam
"""

import torch
import torch.nn.functional as F

from guide_nlp.chapter21.seq2seq import BOS, EOS, PAD, Seq2Seq

Beam = tuple[float, list[int], torch.Tensor]


@torch.no_grad()
def beam(
    model: Seq2Seq, src: torch.Tensor, k: int = 5, alpha: float = 0.7, maxlen: int = 30
) -> list[int]:
    enc_out, s = model.encoder(src)
    mask = src != PAD
    beams: list[Beam] = [(0.0, [BOS], s)]
    finished: list[tuple[float, list[int]]] = []
    for _ in range(maxlen):
        candidates: list[Beam] = []
        for score, seq, state in beams:
            token = torch.tensor([seq[-1]], device=src.device)
            logits, new_state, _ = model.step(token, state, enc_out, mask)
            top = F.log_softmax(logits[0], -1).topk(k)
            for value, i in zip(top.values.tolist(), top.indices.tolist(), strict=True):
                candidates.append((score + value, [*seq, i], new_state))
        candidates.sort(key=lambda c: -c[0])
        beams = []
        for candidate in candidates[:k]:
            if candidate[1][-1] == EOS:
                length = len(candidate[1]) - 1
                finished.append((candidate[0] / length**alpha, candidate[1]))
            else:
                beams.append(candidate)
        if not beams:
            break
    return max(finished or [(b[0], b[1]) for b in beams], key=lambda f: f[0])[1][1:]


@torch.no_grad()
def greedy(model: Seq2Seq, src: torch.Tensor, maxlen: int = 30) -> list[int]:
    """The most probable token at each step (one sentence)."""
    enc_out, s = model.encoder(src)
    mask = src != PAD
    token = torch.tensor([BOS], device=src.device)
    out: list[int] = []
    for _ in range(maxlen):
        logits, s, _ = model.step(token, s, enc_out, mask)
        token = logits.argmax(-1)
        if int(token.item()) == EOS:
            break
        out.append(int(token.item()))
    return out


def main() -> None:
    torch.manual_seed(0)
    model = Seq2Seq(src_vocab=50, tgt_vocab=50).eval()
    src = torch.tensor([[5, 6, 7, 8]])
    print("greedy:", greedy(model, src, maxlen=8))
    print("beam 5:", beam(model, src, maxlen=8))  # an untrained model: random tokens


if __name__ == "__main__":
    main()
