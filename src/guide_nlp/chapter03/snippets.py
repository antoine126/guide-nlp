"""The short examples of chapter 3, in the order of the text.

Usage: uv run python -m guide_nlp.chapter03.snippets
"""

import regex


def graphemes() -> None:
    """One visible character can be several code points (§ graphemes)."""
    family = "\U0001f468‍\U0001f469‍\U0001f467"  # man ZWJ woman ZWJ girl
    flag = "\U0001f1eb\U0001f1f7"  # regional indicators F, R

    for text in (family, flag, "café"):
        clusters = regex.findall(r"\X", text)
        print(len(text), len(clusters))
    # 5 1
    # 2 1
    # 5 4


def invisible_spaces() -> None:
    """Spaces that are not U+0020."""
    text = "Prix : 12 €"  # 'Prix : 12 €' with no-break spaces
    print("Prix :" in text)  # False: U+00A0 is not U+0020
    print(text.split())  # ['Prix', ':', '12', '€']
    print("​".isspace())  # False: zero-width space is no space


def mojibake() -> None:
    """Text decoded with the wrong codec can be repaired by reversing the mistake."""
    broken = "cafÃ©"
    print(broken.encode("cp1252").decode("utf-8"))  # café


def main() -> None:
    for demo in (graphemes, invisible_spaces, mojibake):
        print(f"--- {demo.__name__}")
        demo()


if __name__ == "__main__":
    main()
