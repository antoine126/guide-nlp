"""The same crawl as a Scrapy spider (chapter 9).

Usage (with the local site running: uv run python -m guide_nlp.chapter09.local_site):
    uv run --group web scrapy runspider src/guide_nlp/chapter09/article_spider.py \\
        -a start_url=http://127.0.0.1:8765/articles -O data/interim/articles.jsonl
"""

from collections.abc import Iterator
from typing import Any

import scrapy
from scrapy.http import Response, TextResponse

from guide_nlp.config import settings


class ArticleSpider(scrapy.Spider):
    name = "articles"
    custom_settings = {  # noqa: RUF012 (Scrapy reads it as a plain class attribute)
        "USER_AGENT": settings.user_agent,
        "ROBOTSTXT_OBEY": True,  # honour robots.txt
        "AUTOTHROTTLE_ENABLED": True,  # adapt speed to server latency
        "CONCURRENT_REQUESTS_PER_DOMAIN": 4,
        "HTTPCACHE_ENABLED": True,  # keep raw responses on disk
    }

    def __init__(self, start_url: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.start_urls = [start_url]

    def parse(self, response: Response, *args: Any, **kwargs: Any) -> Iterator[Any]:
        assert isinstance(response, TextResponse)  # HTML pages: css() is available
        yield from response.follow_all(css="a.lien", callback=self.parse_article)
        yield from response.follow_all(css="a[rel=next]", callback=self.parse)

    def parse_article(self, response: Response, *args: Any, **kwargs: Any) -> Iterator[dict]:
        assert isinstance(response, TextResponse)
        yield {
            "url": response.url,
            "title": response.css("article h1::text").get(),
            "date": response.css("p.date::text").get(),
        }
