"""Canonical URLs, to avoid fetching the same page twice (chapter 9).

Usage: uv run python -m guide_nlp.chapter09.urls
"""

from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

TRACKING = {"utm_source", "utm_medium", "utm_campaign", "fbclid", "gclid"}


def canonical_url(url: str) -> str:
    """Map equivalent URLs to a single canonical form."""
    parts = urlsplit(url)
    query = sorted((k, v) for k, v in parse_qsl(parts.query) if k not in TRACKING)
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, urlencode(query), ""))


if __name__ == "__main__":
    print(canonical_url("HTTPS://Exemple.FR/a/?utm_source=x&b=2&a=1#haut"))
    # https://exemple.fr/a?a=1&b=2
