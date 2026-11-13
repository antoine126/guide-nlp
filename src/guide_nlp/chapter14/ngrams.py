"""Bigram language models: counting, perplexity, Kneser-Ney smoothing (chapter 14).

Usage: uv run --group classic python -m guide_nlp.chapter14.ngrams
"""

import math
from collections import Counter
from collections.abc import Callable
from itertools import pairwise

from guide_nlp.chapter14.data import sentences
from guide_nlp.corpus.load import load_articles

BOS, EOS, UNK = "<s>", "</s>", "<unk>"
LogProb = Callable[[str, str], float]


def seq(sentence: list[str], vocab: set[str]) -> list[str]:
    return [BOS] + [t if t in vocab else UNK for t in sentence] + [EOS]


def count_bigrams(
    sentences: list[list[str]], vocab: set[str]
) -> tuple[Counter[tuple[str, str]], Counter[str]]:
    bigrams: Counter[tuple[str, str]] = Counter()
    contexts: Counter[str] = Counter()
    for sentence in sentences:
        q = seq(sentence, vocab)
        bigrams.update(pairwise(q))
        contexts.update(q[:-1])
    return bigrams, contexts


def perplexity(sentences: list[list[str]], vocab: set[str], logprob: LogProb) -> float:
    total, n = 0.0, 0
    for sentence in sentences:
        q = seq(sentence, vocab)
        for prev, word in pairwise(q):
            total += logprob(prev, word)
            n += 1
    return math.exp(-total / n)


def kneser_ney(bigrams: Counter, contexts: Counter, d: float = 0.75) -> LogProb:
    types_ending: Counter[str] = Counter()
    types_following: Counter[str] = Counter()
    for prev, word in bigrams:
        types_ending[word] += 1
        types_following[prev] += 1
    n_types = len(bigrams)

    def logprob(prev: str, word: str) -> float:
        continuation = types_ending[word] / n_types
        c = contexts[prev]
        if c == 0:
            return math.log(max(continuation, 1e-12))
        discounted = max(bigrams[(prev, word)] - d, 0) / c
        weight = d * types_following[prev] / c
        return math.log(max(discounted + weight * continuation, 1e-12))

    return logprob


def add_k(bigrams: Counter, contexts: Counter, vocab_size: int, k: float = 1.0) -> LogProb:
    """Laplace (k = 1) and add-k smoothing, for comparison."""

    def logprob(prev: str, word: str) -> float:
        return math.log((bigrams[(prev, word)] + k) / (contexts[prev] + k * vocab_size))

    return logprob


def main() -> None:
    train = list(sentences(load_articles("train")))
    test = list(sentences(load_articles("test")))
    counts = Counter(t for s in train for t in s)
    vocab = {w for w, c in counts.items() if c >= 3}
    bigrams, contexts = count_bigrams(train, vocab)
    size = len(vocab) + 2  # + <unk> and </s>
    for k in (1, 0.01):
        print(f"add-{k} bigram: {perplexity(test, vocab, add_k(bigrams, contexts, size, k)):.1f}")
    print(f"Kneser-Ney bigram: {perplexity(test, vocab, kneser_ney(bigrams, contexts)):.1f}")
    # Book, table 14.1: add-1 1650.4, add-0.01 426.1, Kneser-Ney 219.2


if __name__ == "__main__":
    main()
