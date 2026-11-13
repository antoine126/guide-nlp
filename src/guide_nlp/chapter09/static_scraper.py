"""A paginated listing scraper (chapter 9).

Usage: uv run --group web python -m guide_nlp.chapter09.static_scraper
"""

from collections.abc import Iterator
from urllib.parse import urljoin

import httpx
from selectolax.lexbor import LexborHTMLParser

from guide_nlp.chapter09.common import BASE, HEADERS
from guide_nlp.chapter09.local_site import serve_in_background


def scrape_listing(client: httpx.Client, start_url: str) -> Iterator[str]:
    """Follow the 'next' links and yield the URL of every article."""
    url: str | None = start_url
    while url:
        tree = LexborHTMLParser(client.get(url).text)
        for link in tree.css("a.lien"):
            yield urljoin(url, link.attributes["href"])
        next_link = tree.css_first("a[rel=next]")
        url = urljoin(url, next_link.attributes["href"]) if next_link else None


def main() -> None:
    server = serve_in_background()
    with httpx.Client(headers=HEADERS, timeout=10) as client:
        urls = list(scrape_listing(client, f"{BASE}/articles?page=1"))
        print(len(urls), urls[3])  # 50 http://127.0.0.1:8765/article/3
        tree = LexborHTMLParser(client.get(urls[3]).text)
        title, date = tree.css_first("article h1"), tree.css_first("p.date")
        assert title is not None and date is not None
        print(title.text(), date.text())
        # Article numero 3 2026-03-04
    server.shutdown()


if __name__ == "__main__":
    main()
