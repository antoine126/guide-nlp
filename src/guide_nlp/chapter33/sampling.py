"""Decoding knobs on a vector of logits: temperature, top-k, top-p, min-p (chapter 33).

Usage: uv run --group neural python -m guide_nlp.chapter33.sampling
"""

import torch


def apply_temperature(logits: torch.Tensor, t: float) -> torch.Tensor:
    return logits / t


def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    threshold = logits.topk(k).values[-1]
    return logits.masked_fill(logits < threshold, float("-inf"))


def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    sorted_logits, order = logits.sort(descending=True)
    cumulative = sorted_logits.softmax(-1).cumsum(-1)
    drop = cumulative > p
    drop[1:] = drop[:-1].clone()  # keep the token that crosses p
    drop[0] = False
    return logits.scatter(0, order[drop], float("-inf"))


def apply_min_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    probs = logits.softmax(-1)
    return logits.masked_fill(probs < p * probs.max(), float("-inf"))


def main() -> None:
    logits = torch.tensor([3.0, 2.5, 1.0, 0.5, -1.0])
    for name, out in (
        ("t=0.5", apply_temperature(logits, 0.5)),
        ("t=2", apply_temperature(logits, 2.0)),
        ("top-k=2", apply_top_k(logits, 2)),
        ("top-p=0.8", apply_top_p(logits, 0.8)),
        ("min-p=0.2", apply_min_p(logits, 0.2)),
    ):
        print(f"{name:<10}", [round(p, 3) for p in out.softmax(-1).tolist()])


if __name__ == "__main__":
    main()
