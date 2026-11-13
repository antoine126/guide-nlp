"""The short examples of chapter 9, in the order of the text.

Usage: uv run --group web python -m guide_nlp.chapter09.snippets [--online]
Without --online, the API walk runs against the local site's JSON endpoint.
"""

import sys
from collections.abc import Callable

import httpx

from guide_nlp.chapter09.common import BASE, HEADERS
from guide_nlp.chapter09.local_site import serve_in_background

API = "https://fr.wikipedia.org/w/api.php"


def walk_category(client: httpx.Client, category: str, handle: Callable[[str], None]) -> None:
    """Follow the MediaWiki continuation cursor until the last page."""
    params = {
        "action": "query",
        "list": "categorymembers",
        "format": "json",
        "cmtitle": category,
        "cmlimit": 500,
    }
    while True:
        data = client.get(API, params=params).json()
        for member in data["query"]["categorymembers"]:
            handle(member["title"])
        if "continue" not in data:
            break
        params |= data["continue"]  # cursor for the next page


def walk_local_api(client: httpx.Client) -> list[dict]:
    """The same pattern on the hidden JSON API of the local site."""
    page: int | None = 1
    items: list[dict] = []
    while page:
        data = client.get(f"{BASE}/api/articles", params={"page": page}).json()
        items += data["items"]
        page = data["next"]
    return items


def main() -> None:
    with httpx.Client(headers=HEADERS, timeout=30) as client:
        if "--online" in sys.argv:
            titles: list[str] = []
            walk_category(client, "Catégorie:Astronomie", titles.append)
            print(len(titles), titles[:3])
        else:
            server = serve_in_background()
            items = walk_local_api(client)
            print(len(items), items[0])  # 50 {'id': 0, 'title': 'Article 0'}
            server.shutdown()


if __name__ == "__main__":
    main()
