"""Regression tests for the RAG system (chapter 41).

Usage: uv run --group neural --group llm pytest src/guide_nlp/chapter41/test_rag.py
(needs Ollama; about ten minutes for the 55 questions)
"""

import json

import pytest

from guide_nlp.chapter41.rag_system import answer, retrieve
from guide_nlp.config import settings

with open(settings.processed_dir / "qa.jsonl", encoding="utf-8") as handle:
    GOLD = [json.loads(line) for line in handle]
ANSWERABLE = [x for x in GOLD if x["answerable"]]
UNANSWERABLE = [x for x in GOLD if not x["answerable"]]

pytestmark = pytest.mark.ollama


def test_recall_does_not_regress():
    hits = [x["source"] in {c.source for c in retrieve(x["question"], 5)} for x in ANSWERABLE]
    assert sum(hits) / len(hits) >= 0.95


def test_answers_do_not_regress():
    good = [
        any(a.lower() in answer(x["question"]).lower() for a in x["answers"]) for x in ANSWERABLE
    ]
    assert sum(good) / len(good) >= 0.75


@pytest.mark.parametrize("item", UNANSWERABLE, ids=lambda x: x["question"][:40])
def test_no_invention_without_source(item):
    assert "je ne sais pas" in answer(item["question"]).lower()
