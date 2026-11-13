"""Typed access to the processed corpora and evaluation sets.

The files live in ``data/processed/`` (see ``data/processed/SOURCES.md``):

- ``articles.jsonl``: 2,923 French Wikipedia articles in six rubrics, split
  80/20 into train and test (chapter 10);
- ``queries.jsonl``: 112 search queries with their relevant article (chapter 13);
- ``qa.jsonl``, ``qa_hard.jsonl``: questions on the articles (chapters 35-41);
- ``rapports.jsonl``, ``qa_nav.jsonl``, ``qa_rapports.jsonl``: the four public
  reports and their questions (chapters 36-41).
"""

import re
from functools import cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, TypeAdapter

from guide_nlp.config import settings
from guide_nlp.corpus.schema import Report

RUBRICS = ["astronomie", "economie", "histoire", "informatique", "medecine", "sport"]
WORD = re.compile(r"[^\W\d_]+")


class WikiArticle(BaseModel):
    """One article of the processed 'Articles' corpus."""

    pageid: int
    revid: int
    title: str
    url: str
    text: str
    licence: str
    rubric: str
    split: Literal["train", "test"]


class Query(BaseModel):
    """A search query and the ids of its relevant articles."""

    query: str
    relevant: list[int]


class Question(BaseModel):
    """A question whose answer is in one article (or in none)."""

    question: str
    answers: list[str]
    source: int | None = None
    answerable: bool = True


class Part(BaseModel):
    answers: list[str]
    source: int


class HardQuestion(BaseModel):
    """A vague, multi-hop or comparison question (chapters 37-39)."""

    kind: str
    question: str
    parts: list[Part]
    original: str | None = None


class ReportQuestion(BaseModel):
    """A question on a report, with the span that answers it.

    The span is given by character offsets in the extracted report text
    (``rapports.jsonl``, built locally): see ``report_passage``.
    """

    question: str
    answers: list[str]
    source: str
    section: str
    start: int
    end: int


def _read[T](path: Path, model: type[T]) -> list[T]:
    adapter = TypeAdapter(model)
    with path.open(encoding="utf-8") as handle:
        return [adapter.validate_json(line) for line in handle if line.strip()]


@cache
def _articles(path: Path) -> tuple[WikiArticle, ...]:
    return tuple(_read(path, WikiArticle))


def load_articles(split: str | None = None, path: Path | None = None) -> list[WikiArticle]:
    """The 'Articles' corpus, optionally restricted to one split."""
    articles = _articles(path or settings.processed_dir / "articles.jsonl")
    return [a for a in articles if split is None or a.split == split]


def load_queries(path: Path | None = None) -> list[Query]:
    return _read(path or settings.processed_dir / "queries.jsonl", Query)


def load_questions(name: str = "qa") -> list[Question]:
    return _read(settings.processed_dir / f"{name}.jsonl", Question)


def load_hard_questions() -> list[HardQuestion]:
    return _read(settings.processed_dir / "qa_hard.jsonl", HardQuestion)


def load_report_questions(name: str = "qa_rapports") -> list[ReportQuestion]:
    return _read(settings.processed_dir / f"{name}.jsonl", ReportQuestion)


def load_reports() -> list[Report]:
    """The four reports (build them first with ``python -m guide_nlp.corpus.rapports``)."""
    return _read(settings.processed_dir / "rapports.jsonl", Report)


@cache
def _report_texts() -> dict[str, str]:
    return {report.source: report.text for report in load_reports()}


def report_passage(question: ReportQuestion) -> str:
    """The reference passage of a report question, whitespace normalized."""
    text = _report_texts()[question.source][question.start : question.end]
    return " ".join(text.split())


def words(text: str) -> list[str]:
    """Lowercased alphabetic tokens: elided forms give 'l', 'd', 'qu'..."""
    return WORD.findall(text.lower())
