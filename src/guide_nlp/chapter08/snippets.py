"""The short examples of chapter 8, in the order of the text.

Usage: uv run --group web python -m guide_nlp.chapter08.snippets
The docling example runs in its own environment: see envs/docling/README.md.
"""

import csv
import io
from pathlib import Path

from docx import Document
from markitdown import MarkItDown

from guide_nlp.corpus.schema import read_articles


def to_number(value: str) -> float:
    """Parse a French-formatted number."""
    return float(value.replace(" ", "").replace(" ", "").replace(",", "."))


def french_csv() -> None:
    """A semicolon-separated file with French number formatting."""
    data = "commune;population;superficie_km2\nKourou;25 685;2 160,0\nCayenne;63 468;23,6\n"
    rows = list(csv.DictReader(io.StringIO(data), delimiter=";"))
    print(rows[0])
    # {'commune': 'Kourou', 'population': '25 685', 'superficie_km2': '2 160,0'}
    print([to_number(r["superficie_km2"]) for r in rows])  # [2160.0, 23.6]


def validated_jsonl() -> None:
    """``read_articles`` keeps the valid lines and reports the others."""
    path = Path("data/interim/articles_demo.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"url": "https://exemple.fr/a", "title": "Argos", "text": "Le lanceur..."}\n'
        '{"url": "https://exemple.fr/b", "title": "", "text": "Sans titre"}\n',
        encoding="utf-8",
    )
    print([a.title for a in read_articles(path)])
    # line 2 rejected: string_too_short
    # ['Argos']


def office_documents() -> None:
    """Word files: headings with python-docx, everything as Markdown with MarkItDown."""
    path = "data/raw/notes/compte_rendu.docx"
    doc = Document(path)
    headings = [
        p.text
        for p in doc.paragraphs
        if p.style is not None and (p.style.name or "").startswith("Heading")
    ]
    print(headings)
    markdown = MarkItDown().convert(path).text_content
    print(markdown)


def main() -> None:
    for demo in (french_csv, validated_jsonl, office_documents):
        print(f"--- {demo.__name__}")
        demo()


if __name__ == "__main__":
    main()
