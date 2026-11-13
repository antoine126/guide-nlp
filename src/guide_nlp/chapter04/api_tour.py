"""A tour of the ``re`` API (chapter 4).

Usage: uv run python -m guide_nlp.chapter04.api_tour
"""

import re

DATE = re.compile(r"(?P<day>\d{2})/(?P<month>\d{2})/(?P<year>\d{4})")


def main() -> None:
    text = "Le 12/03/2024 à Lyon, le 01/04/2024 à Paris."

    print(DATE.match(text))  # None: text does not start with a date
    m = DATE.search(text)
    assert m is not None
    print(m.group(), m.span())  # 12/03/2024 (3, 13)
    print(m["year"], m.groupdict())  # 2024 {'day': '12', 'month': '03', 'year': '2024'}

    print(DATE.findall(text))  # [('12', '03', '2024'), ('01', '04', '2024')]
    print([m.group() for m in DATE.finditer(text)])  # ['12/03/2024', '01/04/2024']

    # Reorder the groups: ISO 8601 dates.
    print(DATE.sub(r"\g<year>-\g<month>-\g<day>", text))
    # Le 2024-03-12 à Lyon, le 2024-04-01 à Paris.

    print(re.split(r"\s*,\s*", "rouge , vert,bleu"))  # ['rouge', 'vert', 'bleu']
    print(re.split(r"(,)", "a,b"))  # ['a', ',', 'b']


if __name__ == "__main__":
    main()
