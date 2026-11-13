"""Main-content extraction with trafilatura (chapter 8).

Usage: uv run --group web python -m guide_nlp.chapter08.main_content
"""

from pathlib import Path

import trafilatura

PAGE = Path("data/raw/samples/page.html")


def main() -> None:
    raw = PAGE.read_text(encoding="utf-8")
    print(trafilatura.extract(raw, output_format="markdown", include_formatting=True))
    # # Lancement réussi pour le satellite Argos
    #
    # Le lanceur a décollé de Kourou à 10 h 15, heure locale, avec à son bord ...
    #
    # Conçu pour surveiller la montée des eaux, l'engin restera en orbite
    # pendant **sept ans**. Il transmettra chaque jour plusieurs téraoctets ...
    #
    # Selon l'agence, la séparation de l'étage supérieur s'est déroulée ...

    meta = trafilatura.extract_metadata(raw)
    assert meta is not None
    print(meta.title, "|", meta.author, "|", meta.date)
    # Lancement réussi pour le satellite Argos | Léa Martin | 2026-03-12


if __name__ == "__main__":
    main()
