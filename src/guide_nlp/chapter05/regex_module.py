"""The third-party ``regex`` module: Unicode properties, lookbehinds, fuzzy matching.

Usage: uv run python -m guide_nlp.chapter05.regex_module
"""

import regex


def main() -> None:
    text = "Émile et Zoë sont arrivés à Saint-Etienne avec M. Dupont et Mme Durand."

    # Capitalised words, whatever the accents.
    print(regex.findall(r"\p{Lu}\p{Ll}+", text))
    # ['Émile', 'Zoë', 'Saint', 'Etienne', 'Dupont', 'Mme', 'Durand']

    # Variable-length lookbehind: a name after a French title.
    print(regex.findall(r"(?<=\b(?:M\.|Mme|Mlle)\s)\p{Lu}\p{Ll}+", text))
    # ['Dupont', 'Durand']

    # Fuzzy matching: tolerate one error (here, the missing accent).
    m = regex.search(r"(?:Saint-Étienne){e<=1}", text)
    assert m is not None
    print(m.group(), m.fuzzy_counts)  # Saint-Etienne (1, 0, 0)


if __name__ == "__main__":
    main()
