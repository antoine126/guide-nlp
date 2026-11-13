"""Extract dates written in French words (chapter 4).

Usage: uv run python -m guide_nlp.chapter04.dates
"""

import re
from datetime import date

MONTHS = [
    "janvier",
    "février",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "août",
    "septembre",
    "octobre",
    "novembre",
    "décembre",
]

DATE_FR = re.compile(
    rf"""
    \b(?P<day>1er|[12]\d|3[01]|0?[1-9])    # 1er, 1..31
    \s+
    (?P<month>{"|".join(MONTHS)})           # month name
    \s+
    (?P<year>\d{{4}})\b                      # four-digit year
    """,
    re.VERBOSE | re.IGNORECASE,
)


def parse_french_dates(text: str) -> list[date]:
    """Return every 'day month year' date written in French."""
    dates = []
    for m in DATE_FR.finditer(text):
        day = 1 if m["day"] == "1er" else int(m["day"])
        month = MONTHS.index(m["month"].lower()) + 1
        dates.append(date(int(m["year"]), month, day))
    return dates


if __name__ == "__main__":
    print(parse_french_dates("Signé le 1er mars 2024, effectif au 15 Avril 2024."))
    # [datetime.date(2024, 3, 1), datetime.date(2024, 4, 15)]
