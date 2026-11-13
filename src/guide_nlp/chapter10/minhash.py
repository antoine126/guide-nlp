"""MinHash from scratch, then MinHash + LSH with datasketch (chapter 10).

Usage: uv run python -m guide_nlp.chapter10.minhash
"""

import random
import re
import zlib
from typing import cast

from datasketch import MinHash, MinHashLSH

PRIME = (1 << 61) - 1


def shingles(text: str, k: int = 3) -> set[str]:
    words = re.findall(r"[^\W\d_]+", text.lower())
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def make_hashes(n: int, seed: int = 0) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    return [(rng.randrange(1, PRIME), rng.randrange(PRIME)) for _ in range(n)]


def signature(items: set[str], hashes: list[tuple[int, int]]) -> list[int]:
    """One minimum per hash function h(x) = (a x + b) mod p."""
    base = [zlib.crc32(s.encode("utf-8")) for s in items]
    return [min((a * x + b) % PRIME for x in base) for a, b in hashes]


def estimate(sig_a: list[int], sig_b: list[int]) -> float:
    return sum(x == y for x, y in zip(sig_a, sig_b, strict=True)) / len(sig_a)


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b)


# --- At scale: datasketch's MinHash and LSH ---------------------------------


def minhash_of(items: set[str]) -> MinHash:
    m = MinHash(num_perm=128, seed=1)
    m.update_batch([s.encode("utf-8") for s in items])
    return m


def drop_near_duplicates(articles: list[dict], threshold: float = 0.5) -> list[dict]:
    """Keep the longest text of each group of near-duplicates."""
    articles = sorted(articles, key=lambda a: -len(a["text"]))
    sets = [shingles(a["text"]) for a in articles]
    lsh, kept = MinHashLSH(threshold=threshold, num_perm=128), []
    for i, (article, items) in enumerate(zip(articles, sets, strict=True)):
        m = minhash_of(items)
        # LSH proposes candidates; the exact Jaccard confirms them.
        if not any(
            len(items & sets[j]) / len(items | sets[j]) >= threshold
            for j in cast(list[int], lsh.query(m))
        ):
            lsh.insert(i, m)
            kept.append(article)
    return kept


def main() -> None:
    a = shingles("Le satellite Argos a été lancé de Kourou mardi matin par une fusée Ariane.")
    b = shingles("Le satellite Argos a été lancé de Kourou mardi soir par une fusée Ariane.")
    hashes = make_hashes(128)
    print(
        f"exact Jaccard {jaccard(a, b):.3f}, "
        f"MinHash estimate {estimate(signature(a, hashes), signature(b, hashes)):.3f}"
    )


if __name__ == "__main__":
    main()
