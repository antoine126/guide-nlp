"""The 'Rapports' corpus: long public PDFs cut by their table of contents (chapter 8).

Usage: uv run --group web python -m guide_nlp.corpus.rapports

Put the four PDFs listed in ``REPORTS`` in ``data/raw/rapports/`` first (see
the README of that folder). Writes ``data/processed/rapports.jsonl``.
"""

import re
from collections import Counter
from itertools import pairwise
from pathlib import Path

import pymupdf
from pydantic import BaseModel

from guide_nlp.chapter08.pdf_order import read_two_columns
from guide_nlp.config import settings
from guide_nlp.corpus.schema import Report, SpanSection

REPORTS = {
    "insee_portrait_social_2023.pdf": "France, portrait social, édition 2023 (Insee)",
    "drees_depenses_sante_2024.pdf": "Les dépenses de santé en 2024, édition 2025 (Drees)",
    "anssi_panorama_2024.pdf": "Panorama de la cybermenace 2024 (ANSSI)",
    "anssi_panorama_2025.pdf": "Panorama de la cybermenace 2025 (ANSSI)",
}


class Section(BaseModel):
    """One section of a report, located by its table of contents."""

    source: str
    title: str
    level: int
    start_page: int
    end_page: int
    text: str


def split_by_toc(pdf_path: Path) -> list[Section]:
    """Split a PDF into sections using its embedded table of contents."""
    doc = pymupdf.open(pdf_path)
    toc = doc.get_toc()  # [[level, title, page], ...]
    sections = []
    for i, (level, title, start) in enumerate(toc):
        end = toc[i + 1][2] - 1 if i + 1 < len(toc) else doc.page_count
        end = max(end, start)
        text = "\n".join(read_two_columns(doc[p - 1]) for p in range(start, end + 1))
        sections.append(
            Section(
                source=pdf_path.name,
                title=title,
                level=level,
                start_page=start,
                end_page=end,
                text=text,
            )
        )
    return sections


# --- The full-text version used from chapter 36 on --------------------------


def clean(text: str) -> str:
    """Unify spaces and dashes, rejoin words hyphenated at a line break."""
    text = text.replace("\xa0", " ").replace(" ", " ").replace("‑", "-")
    text = re.sub(r"(?<=[a-zéèêàùîôç])-\n(?=[a-zéèêàùîôç])", "", text)
    return re.sub(r"[ \t]+", " ", text)


def strip_running_lines(pages: list[str]) -> list[str]:
    """Headers and footers are short lines repeated on many pages, digits aside."""

    def key(line: str) -> str:
        return re.sub(r"\d+", "#", line.strip())

    counts: Counter[str] = Counter()
    for page in pages:
        counts.update({key(line) for line in page.splitlines() if 0 < len(line.strip()) < 90})
    running = {line for line, n in counts.items() if n >= max(5, 0.15 * len(pages))}
    return [
        "\n".join(line for line in page.splitlines() if key(line) not in running) for page in pages
    ]


def read_report(pdf_path: Path, title: str) -> Report:
    """Full text with page offsets, and sections located by their headings."""
    doc = pymupdf.open(pdf_path)
    pages = strip_running_lines([clean(page.get_text()) for page in doc])
    offsets, position = [], 0
    for page in pages:
        offsets.append(position)
        position += len(page) + 1
    text = "".join(page + "\n" for page in pages)
    sections: list[SpanSection] = []
    stack: list[str] = []
    for level, heading, page in doc.get_toc(simple=True):
        heading = clean(heading).strip()
        if not heading or page < 1:
            continue
        start = offsets[page - 1]
        words = heading.split()
        found = re.search(r"\s+".join(map(re.escape, words[:8])), pages[page - 1])
        if found:
            start += found.start()
        stack = [*stack[: level - 1], heading]
        sections.append(
            SpanSection(title=heading, level=level, path=list(stack), page=page, start=start)
        )
    sections.sort(key=lambda s: s.start)
    for current, following in pairwise(sections):
        current.end = following.start
    if sections:
        sections[-1].end = len(text)
    sections = [s for s in sections if s.end > s.start]
    return Report(
        source=pdf_path.name,
        title=title,
        n_pages=doc.page_count,
        page_offsets=offsets,
        text=text,
        sections=sections,
    )


def main() -> None:
    folder = settings.raw_dir / "rapports"
    output = settings.processed_dir / "rapports.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as sink:
        for name, title in REPORTS.items():
            report = read_report(folder / name, title)
            sink.write(report.model_dump_json() + "\n")
            sizes = sorted(len(report.section_text(s).split()) for s in report.sections)
            print(
                f"{name}: {report.n_pages} pages, {len(report.text.split())} words, "
                f"{len(sizes)} sections, median {sizes[len(sizes) // 2]} words per section"
            )


if __name__ == "__main__":
    main()
