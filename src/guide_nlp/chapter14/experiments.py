"""The measurements of chapter 14: perplexities, LSA search, LDA and NMF topics.

Usage: uv run --group classic python -m guide_nlp.chapter14.experiments
"""

import math
from collections import Counter, defaultdict
from collections.abc import Callable
from itertools import pairwise

import numpy as np
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel
from sklearn.decomposition import NMF, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from guide_nlp.chapter13.metrics import mean_metrics, rank_metrics
from guide_nlp.chapter14.data import content_words, sentences
from guide_nlp.chapter14.ngrams import BOS, EOS, UNK
from guide_nlp.corpus.load import load_articles, load_queries


def perplexities() -> None:
    """Unigram, bigram and trigram models with several smoothings."""
    train = list(sentences(load_articles("train")))
    test = list(sentences(load_articles("test")))
    counts = Counter(t for s in train for t in s)
    vocab = {w for w, c in counts.items() if c >= 3}

    def seq(s: list[str]) -> list[str]:
        return [BOS] + [t if t in vocab else UNK for t in s] + [EOS]

    uni: Counter[str] = Counter()
    bi: Counter[tuple[str, str]] = Counter()
    tri: Counter[tuple[str, str, str]] = Counter()
    for s in train:
        q = seq(s)
        uni.update(q[1:])
        bi.update(pairwise(q))
        tri.update(zip([BOS, *q], q, q[1:], strict=False))
    ctx1: Counter[str] = Counter()
    for (a, _), c in bi.items():
        ctx1[a] += c
    ctx2: Counter[tuple[str, str]] = Counter()
    for (a, b, _), c in tri.items():
        ctx2[(a, b)] += c
    n_tokens, size = sum(uni.values()), len(vocab) + 2
    ending = Counter(w for _, w in bi)
    following = Counter(p for p, _ in bi)
    following2 = Counter((a, b) for a, b, _ in tri)

    def kn2(p: str, w: str, d: float = 0.75) -> float:
        cont = ending[w] / len(bi)
        c = ctx1[p]
        if c == 0:
            return max(cont, 1e-12)
        return max(max(bi[(p, w)] - d, 0) / c + d * following[p] / c * cont, 1e-12)

    def kn3(p: str, pp: str, w: str, d: float = 0.75) -> float:
        c = ctx2[(pp, p)]
        lower = kn2(p, w)
        if c == 0:
            return lower
        return max(max(tri[(pp, p, w)] - d, 0) / c + d * following2[(pp, p)] / c * lower, 1e-12)

    def interpolated(p: str, pp: str, w: str) -> float:
        p3 = tri[(pp, p, w)] / ctx2[(pp, p)] if ctx2[(pp, p)] else 0
        p2 = bi[(p, w)] / ctx1[p] if ctx1[p] else 0
        return max(0.6 * p3 + 0.3 * p2 + 0.1 * uni[w] / n_tokens, 1e-12)

    def run(prob: Callable[[str, str, str], float]) -> tuple[float, float]:
        total, n, zero = 0.0, 0, 0
        for s in test:
            q = seq(s)
            q2 = [BOS, *q]
            for i in range(1, len(q)):
                value = prob(q[i - 1], q2[i - 1], q[i])
                n += 1
                if value <= 0:
                    zero += 1
                else:
                    total += math.log(value)
        return (math.inf if zero else math.exp(-total / n)), zero / n

    models: dict[str, Callable[[str, str, str], float]] = {
        "MLE unigram": lambda p, pp, w: uni[w] / n_tokens,
        "MLE bigram": lambda p, pp, w: bi[(p, w)] / ctx1[p] if ctx1[p] else 0.0,
        "MLE trigram": lambda p, pp, w: tri[(pp, p, w)] / ctx2[(pp, p)] if ctx2[(pp, p)] else 0.0,
        "add-1 bigram": lambda p, pp, w: (bi[(p, w)] + 1) / (ctx1[p] + size),
        "add-0.01 bigram": lambda p, pp, w: (bi[(p, w)] + 0.01) / (ctx1[p] + 0.01 * size),
        "Kneser-Ney bigram": lambda p, pp, w: kn2(p, w),
        "interpolated trigram .6/.3/.1": interpolated,
        "Kneser-Ney trigram": kn3,
    }
    for name, prob in models.items():
        value, unseen = run(prob)
        print(f"{name:<31} perplexity={value:8.1f}  unseen events={unseen:.1%}")


def lsa_search() -> None:
    # Train then test, as in the book: randomized SVD is approximate, and the
    # row order shifts the scores of LSA by about 0.01.
    articles, queries = load_articles("train") + load_articles("test"), load_queries()
    ids = [a.pageid for a in articles]
    tfidf = TfidfVectorizer(
        tokenizer=content_words, token_pattern=None, sublinear_tf=True, min_df=2
    )
    D = tfidf.fit_transform([a.text for a in articles])

    def run(score: Callable[[str], np.ndarray]) -> dict[str, float]:
        rows = [
            rank_metrics([ids[k] for k in np.argsort(-score(q.query))[:100]], set(q.relevant))
            for q in queries
        ]
        return mean_metrics(rows)

    m = run(lambda q: (D @ tfidf.transform([q]).T).toarray().ravel())
    print(f"TF-IDF      nDCG@10={m['ndcg']:.3f}")
    for k in (100, 400, 800):
        svd = TruncatedSVD(n_components=k, random_state=0)
        doc_vectors = normalize(svd.fit_transform(D))
        m = run(
            lambda q, s=svd, Z=doc_vectors: Z @ normalize(s.transform(tfidf.transform([q]))).ravel()
        )
        print(
            f"LSA k={k:<4} nDCG@10={m['ndcg']:.3f} "
            f"(explained variance {svd.explained_variance_ratio_.sum():.3f})"
        )


def topics() -> None:
    train = load_articles("train")
    docs = [[w for w in content_words(a.text) if len(w) > 2] for a in train]
    dictionary = Dictionary(docs)  # docs: lists of tokens, no stop words
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    bow = [dictionary.doc2bow(doc) for doc in docs]
    for k in (6, 10):
        lda = LdaModel(
            bow,
            id2word=dictionary,
            num_topics=k,
            passes=5,
            alpha="auto",
            eta="auto",
            chunksize=500,
            random_state=0,
        )
        coherence = CoherenceModel(
            model=lda, texts=docs, dictionary=dictionary, coherence="c_v", processes=1
        ).get_coherence()
        print(f"LDA k={k}: c_v={coherence:.3f}")
        if k == 6:
            table: defaultdict[int, Counter[str]] = defaultdict(Counter)
            for article, doc in zip(train, bow, strict=True):
                topic = max(
                    lda.get_document_topics(doc, minimum_probability=0), key=lambda x: x[1]
                )[0]
                table[topic][article.rubric] += 1
            for i in range(k):
                print(
                    f"  topic {i}: {[w for w, _ in lda.show_topic(i, 8)]} {table[i].most_common(2)}"
                )

    tfidf = TfidfVectorizer(
        tokenizer=lambda x: [w for w in content_words(x) if len(w) > 2],
        token_pattern=None,
        sublinear_tf=True,
        min_df=5,
        max_df=0.5,
    )
    X = tfidf.fit_transform(a.text for a in train)
    terms = tfidf.get_feature_names_out()
    nmf = NMF(n_components=6, init="nndsvd", random_state=0, max_iter=300).fit(X)
    for i, component in enumerate(nmf.components_):
        print(f"NMF {i}: {[terms[j] for j in np.argsort(-component)[:8]]}")


def main() -> None:
    perplexities()
    lsa_search()
    topics()


if __name__ == "__main__":
    main()
