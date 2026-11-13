"""An HTTP request with httpx, done properly (chapter 9).

Usage: uv run --group web python -m guide_nlp.chapter09.http_basics
(starts the local test site in the background)
"""

import httpx

from guide_nlp.chapter09.common import BASE, HEADERS
from guide_nlp.chapter09.local_site import serve_in_background


def main() -> None:
    server = serve_in_background()
    # A Client reuses TCP connections: faster for the site and for you.
    with httpx.Client(headers=HEADERS, timeout=10, follow_redirects=True) as client:
        response = client.get(f"{BASE}/articles", params={"page": 1})
        print(response.status_code, response.headers["content-type"], response.encoding)
        # 200 text/html; charset=utf-8 utf-8
        response.raise_for_status()  # raises for 4xx and 5xx
        html = response.text  # decoded with the declared encoding
        print(len(html), "characters")
    server.shutdown()


if __name__ == "__main__":
    main()
