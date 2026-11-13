"""The short examples of chapter 15, in the order of the text.

Usage: uv run --group classic python -m guide_nlp.chapter15.snippets
"""

from collections.abc import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import LinearSVC

from guide_nlp.chapter15.baselines import TFIDF
from guide_nlp.chapter15.data import rubric_split


def cross_validation(train_texts: list[str], train_labels: list[str]) -> None:
    """Five folds give a mean and a spread; the grid search tunes on the folds only."""
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    scores = cross_val_score(
        make_pipeline(TfidfVectorizer(**TFIDF), LinearSVC()),
        train_texts,
        train_labels,
        cv=cv,
        scoring="f1_macro",
    )
    print(np.round(scores, 3), f"mean={scores.mean():.3f} std={scores.std():.3f}")
    # [0.933 0.951 0.963 0.958 0.942] mean=0.949 std=0.011

    pipeline = make_pipeline(TfidfVectorizer(**TFIDF), LogisticRegression(max_iter=2000))
    grid = GridSearchCV(
        pipeline,
        {
            "logisticregression__C": [0.1, 1, 10, 100],
            "tfidfvectorizer__ngram_range": [(1, 1), (1, 2)],
        },
        cv=cv,
        scoring="f1_macro",
        n_jobs=4,
    )
    grid.fit(train_texts, train_labels)
    print(grid.best_params_, round(grid.best_score_, 3))


def coefficients(pipeline: Pipeline) -> None:
    """The ten heaviest words of each rubric in a fitted TF-IDF + linear SVM."""
    vectorizer = pipeline.named_steps["tfidfvectorizer"]
    classifier = pipeline.named_steps["linearsvc"]
    terms = vectorizer.get_feature_names_out()
    for label, coef in zip(classifier.classes_, classifier.coef_, strict=True):
        print(label, [terms[i] for i in np.argsort(-coef)[:10]])


def lexicon_score(tokens: Sequence[str], lexicon: dict[str, int]) -> int:
    """Sum of the polarities (+1 / -1), with a naive negation window."""
    score, negate = 0, 0
    for token in tokens:
        if token in {"pas", "jamais", "plus", "guère"}:
            negate = 3
            continue
        polarity = lexicon.get(token, 0)
        score += -polarity if negate else polarity
        negate = max(negate - 1, 0)
    return score


def main() -> None:
    train_texts, train_labels, _, _ = rubric_split()
    cross_validation(train_texts, train_labels)
    svm = make_pipeline(TfidfVectorizer(**TFIDF), LinearSVC()).fit(train_texts, train_labels)
    coefficients(svm)
    lexicon = {"excellent": 1, "rapide": 1, "bon": 1, "lent": -1, "cassé": -1, "décevant": -1}
    for review in ("livraison rapide et produit excellent", "pas bon du tout , très décevant"):
        print(lexicon_score(review.split(), lexicon), "|", review)


if __name__ == "__main__":
    main()
