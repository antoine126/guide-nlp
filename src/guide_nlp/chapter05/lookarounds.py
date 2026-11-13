"""Lookahead and lookbehind assertions (chapter 5).

Usage: uv run python -m guide_nlp.chapter05.lookarounds
"""

import re


def main() -> None:
    text = "Prix : 349 € (au lieu de 399 €), garantie 2 ans, réf. 4471."

    # Numbers followed by a euro sign, without capturing the sign.
    print(re.findall(r"\d+(?=\s?€)", text))  # ['349', '399']

    # Numbers NOT followed by a euro sign (nor by another digit).
    print(re.findall(r"\b\d+\b(?!\s?€)", text))  # ['2', '4471']

    # A product reference: digits preceded by "réf. ".
    print(re.findall(r"(?<=réf\. )\d+", text))  # ['4471']

    # "plus" not preceded by "ne " (fixed-width negative lookbehind).
    for s in ("je ne veux plus", "plus jamais ça", "deux plus deux"):
        print(s, "->", bool(re.search(r"(?<!\bne )\bplus\b", s)))
    # je ne veux plus -> True   (preceded by "veux ", not "ne ")
    # plus jamais ça -> True
    # deux plus deux -> True


if __name__ == "__main__":
    main()
