"""The word vocabulary shared by the neural chapters (16 to 20)."""

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<s>", "</s>", "<unk>"]


@dataclass(frozen=True)
class Vocab:
    """Index <-> word mappings, with four special tokens first."""

    itos: list[str]

    @classmethod
    def build(cls, sentences: Iterable[list[str]], min_count: int = 3) -> "Vocab":
        counts = Counter(w for s in sentences for w in s)
        return cls(SPECIALS + [w for w, c in counts.most_common() if c >= min_count])

    @property
    def stoi(self) -> dict[str, int]:
        return {w: i for i, w in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: list[str]) -> list[int]:
        stoi = self.stoi
        return [stoi.get(t, UNK) for t in tokens]
