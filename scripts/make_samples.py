"""Build the small sample documents of chapter 8 (a two-column PDF, a Word file).

Usage: uv run --group web python scripts/make_samples.py
Writes data/raw/samples/rapport.pdf and data/raw/notes/compte_rendu.docx.
"""

from pathlib import Path

import pymupdf
from docx import Document

SAMPLES = Path("data/raw/samples")
NOTES = Path("data/raw/notes")

LEFT = (
    "La première colonne commence ici. Elle décrit le contexte du rapport et les "
    "objectifs de la mission d'évaluation."
)
RIGHT = (
    "La seconde colonne présente les résultats. Les dépenses ont augmenté de 4 % "
    "en un an, portées par les soins hospitaliers et par les médicaments. Les "
    "établissements les plus petits concentrent l'essentiel des difficultés de "
    "trésorerie relevées par la mission."
)


def make_pdf(path: Path) -> None:
    """An A4 page: full-width heading, two text columns, a page number in the footer."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((50, 90), "1. Introduction", fontsize=14, fontname="hebo")
    page.insert_textbox(pymupdf.Rect(50, 110, 290, 400), LEFT, fontsize=10, fontname="helv")
    page.insert_textbox(pymupdf.Rect(305, 110, 545, 400), RIGHT, fontsize=10, fontname="helv")
    page.insert_text((294, 815), "1", fontsize=8, fontname="helv")
    doc.set_toc([[1, "1. Introduction", 1]])
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(path)


def make_docx(path: Path) -> None:
    """A short meeting report with two heading levels."""
    doc = Document()
    doc.add_heading("Compte rendu de la réunion du 12 mars 2026", level=1)
    doc.add_paragraph("Présents : L. Martin, A. Dubois, K. Nguyen.")
    doc.add_heading("Ordre du jour", level=2)
    doc.add_paragraph("Bilan du lancement du satellite Argos et calendrier des premières images.")
    doc.add_heading("Décisions", level=2)
    doc.add_paragraph("La diffusion des premières images est fixée au 25 mars.")
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))


if __name__ == "__main__":
    make_pdf(SAMPLES / "rapport.pdf")
    make_docx(NOTES / "compte_rendu.docx")
    print("written:", SAMPLES / "rapport.pdf", NOTES / "compte_rendu.docx")
