"""The French patterns of appendix A, on the examples given in their comments."""

import re

import pytest

from guide_nlp.annexes import regex_fr as rx


@pytest.mark.parametrize(
    ("pattern", "text", "expected"),
    [
        (rx.WORD, "L'été à Saint-Étienne", ["L'été", "à", "Saint-Étienne"]),
        (
            rx.LONG_DATE,
            "le 1er mai 2024 et le 14 juillet 1789",
            ["1er mai 2024", "14 juillet 1789"],
        ),
        (rx.PHONE, "01 23 45 67 89 ou +33 6 12 34 56 78", ["01 23 45 67 89", "+33 6 12 34 56 78"]),
        (rx.POSTCODE, "75011, 97400, pas 00123", ["75011", "97400"]),
        (rx.AMOUNT, "Total : 1 234,56 €", ["1 234,56 €"]),
        (rx.EMAIL, "écrire à marie.durand+test@exemple.fr.", ["marie.durand+test@exemple.fr"]),
        (rx.URL, "(voir https://exemple.fr/doc)", ["https://exemple.fr/doc"]),
        (rx.IBAN, "IBAN FR76 3000 6000 0112 3456 7890 189", ["FR76 3000 6000 0112 3456 7890 189"]),
    ],
)
def test_examples(pattern: str, text: str, expected: list[str]) -> None:
    assert [m.group() for m in re.finditer(pattern, text)] == expected


def test_sentences() -> None:
    assert re.split(rx.SENTENCE_END, "Il pleut. Il fait froid ! « Rentrons. »") == [
        "Il pleut.",
        "Il fait froid !",
        "« Rentrons. »",
    ]
