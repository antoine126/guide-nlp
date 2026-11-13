"""Analogies a:b :: c:? on the corpus vectors and on frWac (chapter 17).

Usage: uv run --group classic python -m guide_nlp.chapter17.analogies
"""

from itertools import permutations

from gensim.models import KeyedVectors

from guide_nlp.chapter17.corpus import FRWAC, load_frwac, train_sentences
from guide_nlp.chapter17.train_gensim import train

GROUPS = {
    "capitales": [
        ("france", "paris"),
        ("italie", "rome"),
        ("espagne", "madrid"),
        ("allemagne", "berlin"),
        ("japon", "tokyo"),
        ("russie", "moscou"),
        ("portugal", "lisbonne"),
        ("grèce", "athènes"),
        ("égypte", "caire"),
        ("chine", "pékin"),
    ],
    "genre": [
        ("roi", "reine"),
        ("homme", "femme"),
        ("oncle", "tante"),
        ("frère", "sœur"),
        ("fils", "fille"),
        ("père", "mère"),
        ("acteur", "actrice"),
        ("prince", "princesse"),
        ("mari", "épouse"),
        ("neveu", "nièce"),
    ],
    "pluriel": [
        ("chat", "chats"),
        ("chien", "chiens"),
        ("livre", "livres"),
        ("ville", "villes"),
        ("maison", "maisons"),
        ("voiture", "voitures"),
        ("enfant", "enfants"),
        ("arbre", "arbres"),
        ("route", "routes"),
    ],
    "participe": [
        ("parler", "parlé"),
        ("chanter", "chanté"),
        ("manger", "mangé"),
        ("donner", "donné"),
        ("trouver", "trouvé"),
        ("marcher", "marché"),
        ("jouer", "joué"),
        ("penser", "pensé"),
        ("regarder", "regardé"),
        ("écouter", "écouté"),
    ],
}


def evaluate_analogies(kv: KeyedVectors) -> tuple[dict[str, tuple[int, int]], float]:
    """Accuracy of b - a + c == d over every ordered pair of pairs of a group."""
    results = {}
    for name, all_pairs in GROUPS.items():
        pairs = [p for p in all_pairs if p[0] in kv and p[1] in kv]
        hits = n = 0
        for (a, b), (c, d) in permutations(pairs, 2):
            predicted = kv.most_similar(positive=[b, c], negative=[a], topn=1)[0][0]
            hits += predicted == d
            n += 1
        results[name] = (hits, n)
    total_hits = sum(h for h, _ in results.values())
    total = sum(n for _, n in results.values())
    return results, total_hits / max(total, 1)


def main() -> None:
    sentences = train_sentences()
    for name, model in (
        ("corpus, skip-gram", train(sentences)),
        ("corpus, CBOW", train(sentences, sg=0)),
    ):
        detail, accuracy = evaluate_analogies(model.wv)
        print(f"{name:<18} {accuracy:.1%} {detail}")
    if FRWAC.exists():
        detail, accuracy = evaluate_analogies(load_frwac())
        print(f"{'frWac':<18} {accuracy:.1%} {detail}")
    # Book: 3.8 % (skip-gram), 1.7 % (CBOW), 68.5 % (frWac)


if __name__ == "__main__":
    main()
