"""Removing the running headers and footers of a PDF (chapter 36)."""

import re
from collections import Counter


def strip_running_lines(pages: list[str]) -> list[str]:
    """Short lines repeated on many pages, digits aside, are headers or footers."""
    counts: Counter[str] = Counter()
    for page in pages:
        counts.update(
            {
                re.sub(r"\d+", "#", line.strip())
                for line in page.splitlines()
                if 0 < len(line.strip()) < 90
            }
        )
    running = {line for line, n in counts.items() if n >= max(5, 0.15 * len(pages))}
    return [
        "\n".join(
            line for line in page.splitlines() if re.sub(r"\d+", "#", line.strip()) not in running
        )
        for page in pages
    ]
