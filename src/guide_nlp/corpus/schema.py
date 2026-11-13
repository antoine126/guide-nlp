"""Validated records of the 'Articles' corpus (chapter 8)."""

from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError


class Article(BaseModel):
    """One document of the 'Articles' corpus."""

    url: str
    title: str = Field(min_length=1)
    published: str | None = None
    text: str


def read_articles(path: Path) -> Iterator[Article]:
    """Stream valid articles from a JSONL file, report the others."""
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                yield Article.model_validate_json(line)
            except ValidationError as err:
                print(f"line {line_number} rejected: {err.errors()[0]['type']}")


# --- The 'Rapports' corpus (chapter 8, used from chapter 36 on) --------------


class SpanSection(BaseModel):
    """A section located by character offsets in the report's full text."""

    title: str
    level: int
    path: list[str]
    page: int
    start: int
    end: int = 0


class Report(BaseModel):
    """A whole report: its text, page offsets and table of contents."""

    source: str
    title: str
    n_pages: int
    page_offsets: list[int]
    text: str
    sections: list[SpanSection]

    def section_text(self, section: SpanSection) -> str:
        return self.text[section.start : section.end]
