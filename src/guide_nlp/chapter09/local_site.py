"""A tiny local website with controlled latency, to test the scrapers of chapter 9.

Usage: uv run python -m guide_nlp.chapter09.local_site [PORT] [LATENCY_S]
(default: port 8765, 0.1 s of latency per request)

Pages: /articles?page=N (5 listing pages of 10 links), /article/<i>,
/api/articles?page=N (the same data as JSON), /dynamique (filled by
JavaScript), /flaky (503 twice out of three calls), /robots.txt.
"""

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

PER_PAGE, N_PAGES = 10, 5
ROBOTS = """User-agent: *
Disallow: /prive/
Crawl-delay: 1

User-agent: BadBot
Disallow: /
"""
DYNAMIC = """<html><body><ul id="liste"></ul><script>
fetch('/api/articles?page=1').then(r => r.json()).then(d => {
  document.getElementById('liste').innerHTML =
    d.items.map(a => `<li class="lien">${a.title}</li>`).join('');
});
</script></body></html>"""


def article_html(i: int) -> str:
    return (
        f"<html><head><title>Article {i}</title></head><body>"
        f"<nav><a href='/articles?page=1'>Accueil</a></nav>"
        f"<article><h1>Article numero {i}</h1>"
        f"<p class='date'>2026-03-{(i % 28) + 1:02d}</p>"
        f"<p>Texte de l'article {i}.</p></article></body></html>"
    )


def listing_html(page: int) -> str:
    start = (page - 1) * PER_PAGE
    links = "".join(
        f"<li><a class='lien' href='/article/{start + k}'>Article {start + k}</a></li>"
        for k in range(PER_PAGE)
    )
    nxt = f"<a rel='next' href='/articles?page={page + 1}'>Suivant</a>" if page < N_PAGES else ""
    return f"<html><body><ul>{links}</ul>{nxt}</body></html>"


def make_handler(latency: float) -> type[BaseHTTPRequestHandler]:
    flaky_calls = {"n": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def reply(
            self,
            code: int,
            body: str,
            ctype: str = "text/html; charset=utf-8",
            extra: dict[str, str] | None = None,
        ) -> None:
            data = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            for key, value in (extra or {}).items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # name imposed by http.server
            time.sleep(latency)
            url = urlparse(self.path)
            page = int(parse_qs(url.query).get("page", ["1"])[0])
            if url.path == "/robots.txt":
                self.reply(200, ROBOTS, "text/plain")
            elif url.path == "/articles":
                self.reply(200, listing_html(page))
            elif url.path.startswith("/article/"):
                self.reply(200, article_html(int(url.path.rsplit("/", 1)[1])))
            elif url.path == "/api/articles":
                start = (page - 1) * PER_PAGE
                items = [
                    {"id": start + k, "title": f"Article {start + k}"} for k in range(PER_PAGE)
                ]
                body = {"items": items, "next": page + 1 if page < N_PAGES else None}
                self.reply(200, json.dumps(body), "application/json")
            elif url.path == "/dynamique":
                self.reply(200, DYNAMIC)
            elif url.path == "/flaky":
                flaky_calls["n"] += 1
                if flaky_calls["n"] % 3:
                    self.reply(503, "busy", extra={"Retry-After": "1"})
                else:
                    self.reply(200, "ok")
            else:
                self.reply(404, "not found")

    return Handler


def make_server(port: int = 8765, latency: float = 0.1) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(latency))
    server.daemon_threads = True
    server.request_queue_size = 256
    return server


def serve_in_background(port: int = 8765, latency: float = 0.1) -> ThreadingHTTPServer:
    """Start the site in a daemon thread (for demos and tests); call .shutdown() to stop."""
    server = make_server(port, latency)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    latency = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
    print(f"serving http://127.0.0.1:{port} (latency {latency}s), Ctrl+C to stop")
    make_server(port, latency).serve_forever()
