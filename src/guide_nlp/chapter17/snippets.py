"""The short examples of chapter 17: pretrained frWac vectors.

Usage: uv run --group classic python -m guide_nlp.chapter17.snippets
(download frWac first: see guide_nlp.chapter17.corpus)
"""

from guide_nlp.chapter17.corpus import load_frwac


def main() -> None:
    vectors = load_frwac()
    print(vectors.most_similar(positive=["roi", "femme"], negative=["homme"], topn=3))
    # [('reine', 0.621), ('régente', 0.588), ('épousa', 0.573)]
    pairs = [("chat", "chien"), ("chat", "voiture"), ("voiture", "automobile"), ("bon", "mauvais")]
    for a, b in pairs:
        print(a, b, round(float(vectors.similarity(a, b)), 3))
    # chat chien 0.621 / chat voiture 0.122 / voiture automobile 0.509 / bon mauvais 0.549


if __name__ == "__main__":
    main()
