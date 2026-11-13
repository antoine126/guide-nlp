"""Word2Vec with gensim, and the effect of the window and of CBOW (chapter 17).

Usage: uv run --group classic python -m guide_nlp.chapter17.train_gensim
"""

from typing import Any

from gensim.models import Word2Vec

from guide_nlp.chapter17.corpus import train_sentences

PARAMS: dict[str, Any] = dict(
    vector_size=100,
    window=5,
    min_count=5,
    sg=1,
    negative=5,
    sample=1e-4,
    epochs=10,
    seed=0,
    workers=8,
)


def train(sentences: list[list[str]], **overrides: Any) -> Word2Vec:
    return Word2Vec(sentences, **{**PARAMS, **overrides})


def main() -> None:
    sentences = train_sentences()
    model = Word2Vec(
        sentences,  # an iterable of token lists
        vector_size=100,  # 100-300 is usual
        window=5,  # maximum distance between center and context
        min_count=5,  # ignore rarer words
        sg=1,  # 1: skip-gram, 0: CBOW
        negative=5,  # k, the number of noise words
        sample=1e-4,  # subsampling threshold t
        epochs=10,
        seed=0,
        workers=8,
    )
    print(model.wv.most_similar("planète", topn=5))
    for name, overrides in (
        ("window=2", {"window": 2}),
        ("window=15", {"window": 15}),
        ("CBOW", {"sg": 0}),
    ):
        variant = train(sentences, **overrides)
        for word in ("guerre", "planète"):
            print(f"{name:<9} {word} -> {[w for w, _ in variant.wv.most_similar(word, topn=7)]}")


if __name__ == "__main__":
    main()
