"""The test file of chapter 5: a pattern is code, it deserves tests."""

import pytest

from guide_nlp.text.patterns import NUMERIC_DATE


@pytest.mark.parametrize("text", ["12/03/2024", "1.3.2024", "31-12-1999"])
def test_numeric_date_matches(text: str) -> None:
    assert NUMERIC_DATE.fullmatch(text)


@pytest.mark.parametrize(
    "text", ["32/01/2024", "12/13/2024", "12/03/24", "112/03/2024", "tel 06/12/2024/5"]
)
def test_numeric_date_rejects(text: str) -> None:
    assert not NUMERIC_DATE.fullmatch(text)
