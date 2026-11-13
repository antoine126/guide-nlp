"""A first regular expression (chapter 4).

Usage: uv run python -m guide_nlp.chapter04.first
"""

import re


def main() -> None:
    text = "Commande du 12/03/2024, livrée le 15/03/2024 à Lyon."
    print(re.findall(r"\d{2}/\d{2}/\d{4}", text))
    # ['12/03/2024', '15/03/2024']


if __name__ == "__main__":
    main()
