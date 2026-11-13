"""Business rules applied after extraction: cheap, exact, explainable (chapter 47).

Usage: uv run python -m guide_nlp.chapter47.business_rules
"""

import re


def problems(invoice: dict) -> list[str]:
    found = []
    number, date = str(invoice.get("numero") or ""), str(invoice.get("date") or "")
    if date and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        found.append("date mal formée")
    if number and not re.fullmatch(r"FA-\d{4}-\d{4}", number):
        found.append("numéro mal formé")
    if number and date and number[3:7] != date[:4]:
        found.append("l'année du numéro et celle de la date diffèrent")
    return found


if __name__ == "__main__":
    for invoice in (
        {"numero": "FA-2021-2208", "date": "2021-10-26"},
        {"numero": "FA-2021-2208", "date": "2022-10-26"},
        {"numero": "2208", "date": "26/10/2021"},
    ):
        print(invoice, "->", problems(invoice) or "ok")
