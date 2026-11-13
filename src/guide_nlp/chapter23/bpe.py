"""Byte-pair encoding from scratch (chapter 23).

Usage: uv run python -m guide_nlp.chapter23.bpe
"""

from collections import Counter
from itertools import pairwise


def merge_pair(symbols: tuple[str, ...], pair: tuple[str, str]) -> tuple[str, ...]:
    """Replace every occurrence of `pair` by its concatenation."""
    out, i = [], 0
    while i < len(symbols):
        if i < len(symbols) - 1 and (symbols[i], symbols[i + 1]) == pair:
            out.append(symbols[i] + symbols[i + 1])
            i += 2
        else:
            out.append(symbols[i])
            i += 1
    return tuple(out)


def train_bpe(corpus: dict[str, int], num_merges: int) -> list[tuple[str, str]]:
    """Learn BPE merges from a {word: count} dictionary."""
    words = {(*tuple(word), "</w>"): n for word, n in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs: Counter[tuple[str, str]] = Counter()
        for symbols, n in words.items():
            for pair in pairwise(symbols):
                pairs[pair] += n
        if not pairs:
            break
        best = max(pairs, key=lambda p: pairs[p])
        merges.append(best)
        words = {merge_pair(symbols, best): n for symbols, n in words.items()}
    return merges


def apply_bpe(word: str, merges: list[tuple[str, str]]) -> list[str]:
    """Tokenize a word by replaying the merges in the order they were learned."""
    symbols = (*tuple(word), "</w>")
    for pair in merges:
        symbols = merge_pair(symbols, pair)
    return list(symbols)


def main() -> None:
    corpus = {
        "chanter": 5,
        "chantons": 4,
        "chanteur": 3,
        "chanteuse": 2,
        "parler": 6,
        "parlons": 3,
        "parleur": 2,
        "donner": 4,
        "donnons": 2,
    }
    merges = train_bpe(corpus, 12)
    for word in ["chanter", "parleuse", "donneur", "marcher"]:
        print(word, "->", apply_bpe(word, merges))
    # chanter -> ['chant', 'er</w>']
    # parleuse -> ['parl', 'e', 'u', 's', 'e', '</w>']
    # donneur -> ['d', 'on', 'n', 'e', 'u', 'r</w>']
    # marcher -> ['m', 'a', 'r', 'ch', 'er</w>']


if __name__ == "__main__":
    main()
