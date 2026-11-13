"""XPath queries with lxml, and CSS selectors with selectolax (chapter 8).

Usage: uv run --group web python -m guide_nlp.chapter08.html_xpath
"""

from pathlib import Path

from lxml import html
from selectolax.lexbor import LexborHTMLParser

PAGE = Path("data/raw/samples/page.html")


def with_xpath(raw: str) -> None:
    tree = html.fromstring(raw)
    print(tree.xpath("//article/h1/text()"))  # ['Lancement réussi pour le satellite Argos']
    print(tree.xpath("//span[@class='author']/text()"))  # ['Léa Martin']
    print(len(tree.xpath("//article/p[not(@class)]")))  # 3
    print(tree.xpath("//meta[@property='article:published_time']/@content"))
    # ['2026-03-12T10:30:00+01:00']
    print(tree.xpath("string(//article/p[last()])")[:50])
    # Selon l'agence, la séparation de l'étage supérieur


def with_selectolax(raw: str) -> None:
    tree = LexborHTMLParser(raw)
    author = tree.css_first("span.author")
    assert author is not None
    print(author.text())  # Léa Martin
    print([p.text(strip=True)[:30] for p in tree.css("article p:not(.byline)")])


def main() -> None:
    raw = PAGE.read_text(encoding="utf-8")
    with_xpath(raw)
    with_selectolax(raw)


if __name__ == "__main__":
    main()
