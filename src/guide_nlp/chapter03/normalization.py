"""Unicode normalisation forms, accents and case folding (chapter 3).

Usage: uv run python -m guide_nlp.chapter03.normalization
"""

import unicodedata as ud


def strip_accents(text: str) -> str:
    """Remove combining marks after canonical decomposition."""
    decomposed = ud.normalize("NFD", text)
    return "".join(c for c in decomposed if ud.category(c) != "Mn")


def main() -> None:
    composed = "café"  # é as one code point
    decomposed = "café"  # e + COMBINING ACUTE ACCENT
    print(composed == decomposed)  # False
    print(len(composed), len(decomposed))  # 4 5

    nfc = ud.normalize("NFC", decomposed)
    print(nfc == composed)  # True

    sample = "éﬁ²"  # é, ligature fi, superscript two
    for form in ("NFC", "NFD", "NFKC", "NFKD"):
        points = [f"U+{ord(c):04X}" for c in ud.normalize(form, sample)]
        print(form, points)
    # NFC  ['U+00E9', 'U+FB01', 'U+00B2']
    # NFD  ['U+0065', 'U+0301', 'U+FB01', 'U+00B2']
    # NFKC ['U+00E9', 'U+0066', 'U+0069', 'U+0032']
    # NFKD ['U+0065', 'U+0301', 'U+0066', 'U+0069', 'U+0032']

    print(strip_accents("Hôpital, élève, Noël, cœur"))  # Hopital, eleve, Noel, cœur
    print("Straße".lower(), "Straße".casefold())  # straße strasse


if __name__ == "__main__":
    main()
