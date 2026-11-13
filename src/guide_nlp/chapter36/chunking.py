"""Chunking strategies as character spans: fixed, sentence-based, structural, semantic
(chapter 36).

Usage: uv run --group neural python -m guide_nlp.chapter36.chunking
(needs data/processed/rapports.jsonl: python -m guide_nlp.corpus.rapports)
"""

import re
from collections.abc import Callable

import numpy as np

from guide_nlp.corpus.load import load_reports
from guide_nlp.corpus.schema import Report

Span = tuple[int, int]
SENT = re.compile(r"(?<=[.!?…:;])\s+(?=[A-ZÉÈÀÂÊÎ«\d(•–-])|\n\s*\n")


def word_spans(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def fixed(text: str, size: int, overlap: int = 0) -> list[tuple[int, int]]:
    words = word_spans(text)
    step = size - overlap
    return [
        (words[i][0], words[min(i + size, len(words)) - 1][1])
        for i in range(0, max(len(words) - overlap, 1), step)
    ]


def sentence_spans(text: str, start: int = 0, end: int | None = None) -> list[Span]:
    end = len(text) if end is None else end
    spans, a = [], start
    for m in SENT.finditer(text, start, end):
        if m.start() > a:
            spans.append((a, m.start()))
        a = m.end()
    if end > a and text[a:end].strip():
        spans.append((a, end))
    return spans


def merge(spans: list[Span], text: str, target: int) -> list[Span]:
    """Greedily merge consecutive pieces up to a target size; cut what is still too long."""
    out: list[Span] = []
    current: Span | None = None
    n = 0
    for a, b in spans:
        k = len(text[a:b].split())
        if k > target:
            if current:
                out.append(current)
                current, n = None, 0
            words = [(a + m.start(), a + m.end()) for m in re.finditer(r"\S+", text[a:b])]
            out += [
                (words[i][0], words[min(i + target, len(words)) - 1][1])
                for i in range(0, len(words), target)
            ]
            continue
        if current and n + k > target:
            out.append(current)
            current, n = None, 0
        current = (current[0], b) if current else (a, b)
        n += k
    if current:
        out.append(current)
    return out


def recursive(report: Report, target: int) -> list[Span]:
    """Whole sentences, merged up to the target size."""
    return merge(sentence_spans(report.text), report.text, target)


def structural(report: Report, target: int) -> list[tuple[Span, str]]:
    """Never cross a section; each chunk remembers its path of headings."""
    out = []
    for s in report.sections:
        for span in merge(sentence_spans(report.text, s.start, s.end), report.text, target):
            out.append((span, " > ".join(s.path)))
    return out


def semantic(
    report: Report,
    encode: Callable[[list[str]], np.ndarray],
    percentile: float = 90,
    cap: int = 400,
) -> list[Span]:
    """Cut where consecutive sentences are least similar (or at the size cap)."""
    spans = sentence_spans(report.text)
    vectors = encode([report.text[a:b] for a, b in spans])
    distance = 1 - np.sum(vectors[:-1] * vectors[1:], axis=1)
    threshold = np.percentile(distance, percentile)
    out, start, n = [], spans[0][0], 0
    for i, (a, b) in enumerate(spans):
        n += len(report.text[a:b].split())
        if i == len(spans) - 1 or distance[i] > threshold or n >= cap:
            out.append((start, b))
            if i + 1 < len(spans):
                start = spans[i + 1][0]
            n = 0
    return out


def main() -> None:
    report = load_reports()[0]
    for name, spans in (
        ("fixed 200", fixed(report.text, 200)),
        ("fixed 200 / 40", fixed(report.text, 200, 40)),
        ("sentences, 200", recursive(report, 200)),
        ("sections, 200", [s for s, _ in structural(report, 200)]),
    ):
        sizes = sorted(len(report.text[a:b].split()) for a, b in spans)
        print(f"{name:<16} {len(spans):>5} chunks, median {sizes[len(sizes) // 2]} words")


if __name__ == "__main__":
    main()
