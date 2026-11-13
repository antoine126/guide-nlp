"""Inventory of the non-ASCII characters of a large file (chapter 3).

Usage: uv run python -m guide_nlp.chapter03.inventory [FILE]
(default: the 'Articles' corpus)
"""

import sys
import unicodedata as ud
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

from guide_nlp.config import settings


def iter_lines(path: Path) -> Iterator[str]:
    """Stream a large UTF-8 file line by line (constant memory)."""
    with path.open(encoding="utf-8", errors="replace") as handle:
        yield from handle


def non_ascii_inventory(path: Path, top: int = 30) -> None:
    counts = Counter(c for line in iter_lines(path) for c in line if ord(c) > 127)
    for char, n in counts.most_common(top):
        print(f"U+{ord(char):04X}  {ud.name(char, '<unnamed>'):<45} {n:>8}")


if __name__ == "__main__":
    default = settings.processed_dir / "articles.jsonl"
    non_ascii_inventory(Path(sys.argv[1]) if len(sys.argv) > 1 else default)
