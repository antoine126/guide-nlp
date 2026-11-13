"""Bounded concurrency with asyncio and httpx, and Little's law (chapter 9).

Usage: uv run --group web python -m guide_nlp.chapter09.concurrent
With 0.1 s of latency per request, throughput grows almost linearly with
the number of requests in flight (Little's law: throughput = in flight / latency).
"""

import asyncio
import time

import httpx

from guide_nlp.chapter09.common import BASE, HEADERS
from guide_nlp.chapter09.local_site import serve_in_background


async def fetch_all(urls: list[str], concurrency: int) -> list[str]:
    """Download pages with at most `concurrency` requests in flight."""
    semaphore = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(max_connections=concurrency)
    async with httpx.AsyncClient(headers=HEADERS, timeout=30, limits=limits) as client:

        async def fetch(url: str) -> str:
            async with semaphore:
                response = await client.get(url)
                response.raise_for_status()
                return response.text

        return await asyncio.gather(*(fetch(url) for url in urls))


def main() -> None:
    server = serve_in_background()
    targets = [f"{BASE}/article/{i}" for i in range(200)]
    for concurrency in (1, 2, 5, 10, 20, 50):
        n = {1: 20, 2: 40}.get(concurrency, 200)
        start = time.perf_counter()
        pages = asyncio.run(fetch_all(targets[:n], concurrency))
        elapsed = time.perf_counter() - start
        print(
            f"concurrency={concurrency:<3} n={len(pages):<4} {elapsed:6.2f}s "
            f"throughput={n / elapsed:6.1f} req/s"
        )
    server.shutdown()


if __name__ == "__main__":
    main()
