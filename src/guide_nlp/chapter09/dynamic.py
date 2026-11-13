"""A page filled by JavaScript: raw HTML vs rendered page (chapter 9).

Usage: uv run --group web python -m guide_nlp.chapter09.dynamic
(first time: uv run --group web playwright install chromium)
"""

import httpx
from playwright.sync_api import Response, sync_playwright
from selectolax.lexbor import LexborHTMLParser

from guide_nlp.chapter09.local_site import serve_in_background

URL = "http://127.0.0.1:8766/dynamique"


def main() -> None:
    server = serve_in_background(port=8766)
    raw = httpx.get(URL).text
    print("raw HTML:", len(LexborHTMLParser(raw).css("li.lien")), "items")
    # raw HTML: 0 items

    api_calls: list[str] = []

    def record(response: Response) -> None:
        if "json" in response.headers.get("content-type", ""):
            api_calls.append(response.url)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.on("response", record)
        page.goto(URL, wait_until="networkidle")
        page.wait_for_selector("li.lien")
        rendered = page.content()
        browser.close()

    print("rendered HTML:", len(LexborHTMLParser(rendered).css("li.lien")), "items")
    # rendered HTML: 10 items
    print("JSON calls seen:", api_calls)
    # JSON calls seen: ['http://127.0.0.1:8766/api/articles?page=1']
    server.shutdown()


if __name__ == "__main__":
    main()
