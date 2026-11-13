"""The measurements of chapter 13: preprocessing, the (k1, b) grid, the misses.

Usage: uv run --group classic python -m guide_nlp.chapter13.experiments
"""

from collections.abc import Callable

from guide_nlp.chapter12.experiments import no_stop, stems, stems_no_stop
from guide_nlp.chapter13.bm25 import BM25Index
from guide_nlp.chapter13.metrics import mean_metrics, rank_metrics
from guide_nlp.corpus.load import Query, load_articles, load_queries, words

PIPELINES: dict[str, Callable[[str], list[str]]] = {
    "minuscules": words,
    "+ mots vides retirés": no_stop,
    "+ racinisation": stems,
    "+ mots vides + racinisation": stems_no_stop,
}


def evaluate(
    index: BM25Index, ids: list[int], tokenize: Callable[[str], list[str]], queries: list[Query]
) -> dict[str, float]:
    rows = [
        rank_metrics([ids[d] for d, _ in index.search(tokenize(q.query), 100)], set(q.relevant))
        for q in queries
    ]
    return mean_metrics(rows)


def main() -> None:
    articles, queries = load_articles(), load_queries()
    texts, ids = [a.text for a in articles], [a.pageid for a in articles]
    indexes = {}
    for name, tokenize in PIPELINES.items():
        indexes[name] = BM25Index([tokenize(t) for t in texts])
        m = evaluate(indexes[name], ids, tokenize, queries)
        print(f"{name:<30} MRR={m['mrr']:.3f} R@10={m['recall']:.3f} nDCG@10={m['ndcg']:.3f}")

    # Tune (k1, b) on half of the queries, report on the other half.
    dev, test = queries[0::2], queries[1::2]
    name = "+ mots vides + racinisation"
    index, tokenize = indexes[name], PIPELINES[name]
    grid = {}
    for k1 in (0.3, 0.6, 0.9, 1.2, 1.5, 2.0, 3.0):
        for b in (0.0, 0.25, 0.5, 0.75, 1.0):
            index.k1, index.b = k1, b
            grid[(k1, b)] = evaluate(index, ids, tokenize, dev)["ndcg"]
    best = max(grid, key=lambda key: grid[key])
    print(f"best (k1, b) on dev: {best}, nDCG@10={grid[best]:.3f}")
    for label, (k1, b) in (("defaults", (1.2, 0.75)), ("tuned", best)):
        index.k1, index.b = k1, b
        print(f"{label:<9} test nDCG@10={evaluate(index, ids, tokenize, test)['ndcg']:.3f}")

    print(
        "tf -> weight with k1=1.2, b=0:",
        [(tf, round(tf * 2.2 / (tf + 1.2), 2)) for tf in (1, 2, 3, 5, 10, 50)],
    )


if __name__ == "__main__":
    main()
