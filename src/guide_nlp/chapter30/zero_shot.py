"""Zero-shot classification: compare each article with a sentence per rubric (chapter 30).

Usage: uv run --group neural python -m guide_nlp.chapter30.zero_shot
"""

import numpy as np
from sklearn.metrics import f1_score

from guide_nlp.chapter28.vectors import SOLON, corpus_vectors, load
from guide_nlp.corpus.load import RUBRICS, load_articles


def main() -> None:
    articles = load_articles()
    doc_vectors, _ = corpus_vectors(SOLON)
    model = load(SOLON)
    names = ["astronomie", "économie", "histoire", "informatique", "médecine", "sport"]
    labels = model.encode(
        [f"query : Cet article parle de {n}." for n in names], normalize_embeddings=True
    )
    test = [i for i, a in enumerate(articles) if a.split == "test"]
    predicted = np.argmax(doc_vectors[test] @ labels.T, axis=1)
    gold = [RUBRICS.index(articles[i].rubric) for i in test]
    print(f"zero-shot macro F1 on the test set = {f1_score(gold, predicted, average='macro'):.3f}")
    # Book: 0.919 with this template and the query prefix


if __name__ == "__main__":
    main()
