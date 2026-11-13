"""A linear-chain CRF: path score, partition function, Viterbi (chapter 20).

Usage: uv run --group neural python -m guide_nlp.chapter20.crf
The demo checks the forward algorithm and Viterbi against brute force.
"""

import itertools

import torch
import torch.nn as nn


class CRF(nn.Module):
    def __init__(self, n_tags: int) -> None:
        super().__init__()
        self.trans = nn.Parameter(torch.zeros(n_tags, n_tags))
        self.start = nn.Parameter(torch.zeros(n_tags))
        self.end = nn.Parameter(torch.zeros(n_tags))

    def path_score(self, emissions: torch.Tensor, tags: torch.Tensor) -> torch.Tensor:
        steps = torch.arange(len(tags), device=tags.device)
        return (
            self.start[tags[0]]
            + emissions[steps, tags].sum()
            + self.trans[tags[:-1], tags[1:]].sum()
            + self.end[tags[-1]]
        )

    def log_partition(self, emissions: torch.Tensor) -> torch.Tensor:
        alpha = self.start + emissions[0]
        for e in emissions[1:]:
            scores = alpha.unsqueeze(1) + self.trans + e.unsqueeze(0)
            alpha = torch.logsumexp(scores, dim=0)
        return torch.logsumexp(alpha + self.end, dim=0)

    def nll(self, emissions: torch.Tensor, tags: torch.Tensor) -> torch.Tensor:
        return self.log_partition(emissions) - self.path_score(emissions, tags)

    @torch.no_grad()
    def viterbi(self, emissions: torch.Tensor) -> list[int]:
        delta, back = self.start + emissions[0], []
        for e in emissions[1:]:
            best, idx = (delta.unsqueeze(1) + self.trans).max(dim=0)
            delta = best + e
            back.append(idx)
        tag = int((delta + self.end).argmax().item())
        path = [tag]
        for idx in reversed(back):
            tag = int(idx[tag].item())
            path.append(tag)
        return path[::-1]


def main() -> None:
    torch.manual_seed(0)
    n_tags, length = 4, 5
    crf = CRF(n_tags)
    with torch.no_grad():
        for p in crf.parameters():
            p.normal_()
    emissions = torch.randn(length, n_tags)
    paths = list(itertools.product(range(n_tags), repeat=length))
    scores = torch.stack([crf.path_score(emissions, torch.tensor(p)) for p in paths])
    print(
        f"{len(paths)} paths; log Z forward={crf.log_partition(emissions).item():.6f} "
        f"brute force={torch.logsumexp(scores, 0).item():.6f}"
    )
    print("Viterbi", crf.viterbi(emissions), "brute force", list(paths[int(scores.argmax())]))


if __name__ == "__main__":
    main()
