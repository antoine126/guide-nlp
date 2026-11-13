"""Collect the 'Articles' corpus from the French Wikipedia (chapter 9).

Usage: uv run --group web python -m guide_nlp.corpus.collect [PER_RUBRIC] [MAX_DEPTH]

Two steps: (1) walk the category tree of each rubric to list page titles,
(2) download the plain-text extract of every page, one request at a time.
The texts are under the CC BY-SA 4.0 licence; each record keeps its URL and
revision id. Set GUIDE_NLP_USER_AGENT to a string that identifies you, as the
Wikimedia API policy asks.
"""

import asyncio
import json
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from guide_nlp.config import settings

API = "https://fr.wikipedia.org/w/api.php"
RUBRICS = {
    "astronomie": "Catégorie:Astronomie",
    "economie": "Catégorie:Économie",
    "histoire": "Catégorie:Histoire",
    "informatique": "Catégorie:Informatique",
    "medecine": "Catégorie:Médecine",
    "sport": "Catégorie:Sport",
}
SKIP_PREFIXES = ("Liste", "Portail", "Chronologie", "Modèle", "Wikipédia")
RETRYABLE = {429, 500, 502, 503, 504}


class TemporaryError(Exception):
    """A response worth retrying (rate limit, server error, maxlag)."""


async def api_get(client: httpx.AsyncClient, params: dict[str, Any]) -> dict[str, Any]:
    """GET the API, retrying temporary failures with exponential backoff."""
    params = {"format": "json", "formatversion": 2, "maxlag": 5, **params}
    retrying = AsyncRetrying(
        retry=retry_if_exception_type((TemporaryError, httpx.TransportError)),
        wait=wait_exponential_jitter(initial=1, max=60),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    async for attempt in retrying:
        with attempt:
            response = await client.get(API, params=params)
            if response.status_code in RETRYABLE:
                raise TemporaryError(f"HTTP {response.status_code}")
            response.raise_for_status()
            data = response.json()
            if data.get("error", {}).get("code") == "maxlag":
                raise TemporaryError("maxlag")
            return data
    raise AssertionError("unreachable")


async def list_titles(
    client: httpx.AsyncClient, root: str, wanted: int, max_depth: int
) -> set[str]:
    """Breadth-first walk of a category tree, collecting page titles."""
    titles: set[str] = set()
    seen = {root}
    queue = deque([(root, 0)])
    while queue and len(titles) < wanted:
        category, depth = queue.popleft()
        params: dict[str, Any] = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmtype": "page|subcat",
            "cmlimit": 500,
        }
        while True:
            data = await api_get(client, params)
            for member in data["query"]["categorymembers"]:
                title = member["title"]
                if member["ns"] == 0 and not title.startswith(SKIP_PREFIXES):
                    titles.add(title)
                elif member["ns"] == 14 and depth < max_depth and title not in seen:
                    seen.add(title)
                    queue.append((title, depth + 1))
            if "continue" not in data:
                break
            params |= data["continue"]
            await asyncio.sleep(0.1)
        await asyncio.sleep(0.1)
    return titles


async def fetch_article(client: httpx.AsyncClient, title: str) -> dict | None:
    """Download the plain text of one page, with its traceability fields."""
    data = await api_get(
        client,
        {
            "action": "query",
            "prop": "extracts|info",
            "explaintext": 1,
            "exsectionformat": "plain",
            "inprop": "url",
            "titles": title,
        },
    )
    page = data["query"]["pages"][0]
    if page.get("missing") or not page.get("extract"):
        return None
    return {
        "pageid": page["pageid"],
        "revid": page["lastrevid"],
        "title": page["title"],
        "url": page["fullurl"],
        "text": page["extract"],
        "licence": "CC BY-SA 4.0",
    }


async def collect(output: Path, per_rubric: int, max_depth: int) -> None:
    """Write one JSON line per article, rubric by rubric."""
    random.seed(42)
    headers = {"User-Agent": settings.user_agent}
    async with httpx.AsyncClient(headers=headers, timeout=30, follow_redirects=True) as client:
        pools: dict[str, set[str]] = {}
        for rubric, root in RUBRICS.items():
            pools[rubric] = await list_titles(client, root, per_rubric * 4, max_depth)
            print(f"{rubric:<14} {len(pools[rubric])} candidate titles", flush=True)
        # A page reachable from two rubrics has an ambiguous label: drop it.
        counts: dict[str, int] = {}
        for titles in pools.values():
            for title in titles:
                counts[title] = counts.get(title, 0) + 1
        ambiguous = sum(1 for n in counts.values() if n > 1)
        print(f"ambiguous titles dropped: {ambiguous}", flush=True)

        start = time.perf_counter()
        n_done = 0
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as sink:
            for rubric, titles in pools.items():
                unique = sorted(t for t in titles if counts[t] == 1)
                random.shuffle(unique)
                for title in unique[:per_rubric]:
                    article = await fetch_article(client, title)
                    if article is not None:
                        article["rubric"] = rubric
                        sink.write(json.dumps(article, ensure_ascii=False) + "\n")
                        n_done += 1
                    await asyncio.sleep(0.05)
                elapsed = time.perf_counter() - start
                print(f"{rubric:<14} done, {n_done} articles, {elapsed:.0f}s", flush=True)


def main() -> None:
    per_rubric = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    output = settings.raw_dir / "wikipedia" / "articles_raw.jsonl"
    asyncio.run(collect(output, per_rubric, max_depth))
    print(f"written: {output}")


if __name__ == "__main__":
    main()
