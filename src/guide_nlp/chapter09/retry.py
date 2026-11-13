"""Retries with exponential backoff and jitter (chapter 9).

Usage: uv run --group web python -m guide_nlp.chapter09.retry
The local /flaky page answers 503 twice out of three calls.
"""

import asyncio

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from guide_nlp.chapter09.common import BASE, HEADERS
from guide_nlp.chapter09.local_site import serve_in_background

RETRYABLE = {429, 500, 502, 503, 504}


class TemporaryError(Exception):
    """A response worth retrying later."""


async def get_with_retries(
    client: httpx.AsyncClient, url: str, multiplier: float = 1.0
) -> httpx.Response:
    retrying = AsyncRetrying(
        retry=retry_if_exception_type((TemporaryError, httpx.TransportError)),
        wait=wait_exponential_jitter(multiplier=multiplier, max=60),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    async for attempt in retrying:
        with attempt:
            response = await client.get(url)
            if response.status_code in RETRYABLE:
                raise TemporaryError(f"HTTP {response.status_code}")
            response.raise_for_status()  # 403, 404: no point in retrying
            return response
    raise AssertionError("unreachable")


async def demo() -> None:
    async with httpx.AsyncClient(headers=HEADERS, timeout=10) as client:
        response = await get_with_retries(client, f"{BASE}/flaky", multiplier=0.1)
        print(response.status_code, response.text)  # 200 ok, after two 503


if __name__ == "__main__":
    server = serve_in_background()
    asyncio.run(demo())
    server.shutdown()
