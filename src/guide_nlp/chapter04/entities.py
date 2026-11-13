"""Phone numbers, e-mail addresses and amounts (chapter 4).

Usage: uv run python -m guide_nlp.chapter04.entities
"""

import re

# 06 12 34 56 78, +33 6 12 34 56 78, 01.23.45.67.89, 0612345678
PHONE_FR = re.compile(r"(?:\+33\s?|\b0)[1-9](?:[\s.-]?\d{2}){4}\b")

# A pragmatic e-mail pattern (RFC 5322 is far more permissive).
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")

# 1 234,50 €, 349 euros, 12€
AMOUNT = re.compile(
    r"(?P<value>\d{1,3}(?:[ .]\d{3})*(?:,\d{1,2})?)\s?(?:€|euros?\b)",
    re.IGNORECASE,
)


def to_float(value: str) -> float:
    """Convert a French-formatted number to a float."""
    return float(value.replace(" ", "").replace(".", "").replace(",", "."))


def anonymize(text: str) -> str:
    """Replace e-mail addresses and phone numbers by placeholders."""
    return PHONE_FR.sub("<TEL>", EMAIL.sub("<EMAIL>", text))


def main() -> None:
    text = "Total : 1 234,50 € TTC, acompte de 300 euros."
    print([to_float(v) for v in AMOUNT.findall(text)])  # [1234.5, 300.0]
    print(anonymize("Écrire à jean.dupont@exemple.fr ou au 06 12 34 56 78."))
    # Écrire à <EMAIL> ou au <TEL>.


if __name__ == "__main__":
    main()
