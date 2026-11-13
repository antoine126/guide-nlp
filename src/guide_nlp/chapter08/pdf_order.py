"""Reading order in PDFs: content stream, spatial sort, columns (chapter 8).

Usage: uv run --group web python -m guide_nlp.chapter08.pdf_order
(the sample PDF is built by scripts/make_samples.py)
"""

from pathlib import Path

import pdfplumber
import pymupdf  # PyMuPDF (formerly imported as "fitz")

SAMPLE = Path("data/raw/samples/rapport.pdf")


def read_two_columns(page: pymupdf.Page, margin: float = 0.05) -> str:
    """Read a two-column page in reading order, without running headers."""
    width, height = page.rect.width, page.rect.height
    top, bottom = height * margin, height * (1 - margin)
    # block = (x0, y0, x1, y1, text, block_no, block_type); type 0 = text
    blocks = [b for b in page.get_text("blocks") if b[6] == 0 and top <= b[1] and b[3] <= bottom]
    middle = width / 2
    left = [b for b in blocks if (b[0] + b[2]) / 2 < middle]
    right = [b for b in blocks if (b[0] + b[2]) / 2 >= middle]
    ordered = sorted(left, key=lambda b: b[1]) + sorted(right, key=lambda b: b[1])
    return "\n".join(b[4].strip() for b in ordered)


def main() -> None:
    doc = pymupdf.open(SAMPLE)
    page = doc[0]

    print(page.get_text())  # content-stream order
    # 1. Introduction
    # La première colonne commence ici. Elle décrit le
    # contexte du rapport et les objectifs de la mission
    # d'évaluation.
    # La seconde colonne présente les résultats. Les ...

    print(page.get_text(sort=True))  # top-to-bottom, left-to-right
    # La première colonne commence ici. Elle décrit le    La seconde colonne ...
    # contexte du rapport et les objectifs de la mission     dépenses ont ...

    with pdfplumber.open(SAMPLE) as pdf:
        print(pdf.pages[0].extract_text())
    # 1. Introduction
    # La première colonne commence ici. Elle décrit le La seconde colonne ...

    print(read_two_columns(page))  # columns in reading order, no page number


if __name__ == "__main__":
    main()
