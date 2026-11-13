"""Parse an HTML page with BeautifulSoup (chapter 8).

Usage: uv run --group web python -m guide_nlp.chapter08.html_bs4
"""

import json
from pathlib import Path

from bs4 import BeautifulSoup, Tag

PAGE = Path("data/raw/samples/page.html")


def main() -> None:
    raw = PAGE.read_text(encoding="utf-8")
    soup = BeautifulSoup(raw, "lxml")  # lxml as the underlying parser

    assert soup.title is not None
    print(soup.title.string)
    # Lancement réussi pour le satellite Argos | Exemple Presse
    author = soup.find("span", class_="author")
    assert isinstance(author, Tag)
    print(author.get_text())  # Léa Martin
    print([a["href"] for a in soup.select("nav.menu a")])  # ['/', '/science', '/sport']

    paragraphs = [p.get_text(" ", strip=True) for p in soup.select("article > p:not(.byline)")]
    print(len(paragraphs), paragraphs[0][:40])
    # 3 Le lanceur a décollé de Kourou à 10 h 15

    # Structured metadata embedded for search engines (JSON-LD).
    script = soup.find("script", type="application/ld+json")
    assert isinstance(script, Tag) and script.string is not None
    ld = json.loads(script.string)
    print(ld["author"]["name"], ld["datePublished"])  # Léa Martin 2026-03-12


if __name__ == "__main__":
    main()
