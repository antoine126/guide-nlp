"""Linear baselines for rubric classification (chapter 15).

Usage: uv run --group classic python -m guide_nlp.chapter15.baselines
"""

from typing import Any

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.svm import LinearSVC

from guide_nlp.chapter15.data import rubric_split

TFIDF: dict[str, Any] = dict(sublinear_tf=True, min_df=2, max_df=0.9)


def baseline_models() -> dict[str, Pipeline]:
    return {
        "NB + counts": make_pipeline(CountVectorizer(min_df=2), MultinomialNB()),
        "NB + TF-IDF": make_pipeline(TfidfVectorizer(**TFIDF), MultinomialNB()),
        "LogReg + TF-IDF": make_pipeline(
            TfidfVectorizer(**TFIDF), LogisticRegression(max_iter=2000)
        ),
        "SVM + TF-IDF": make_pipeline(TfidfVectorizer(**TFIDF), LinearSVC()),
    }


def main() -> None:
    train_texts, train_labels, test_texts, test_labels = rubric_split()
    for name, model in baseline_models().items():
        predicted = model.fit(train_texts, train_labels).predict(test_texts)
        macro_f1 = f1_score(test_labels, predicted, average="macro")
        print(
            f"{name:<16} accuracy={accuracy_score(test_labels, predicted):.3f} "
            f"macro F1={macro_f1:.3f}"
        )
    # Book, table 15.1: NB + counts 0.913 / 0.918, NB + TF-IDF 0.875 / 0.881,
    # LogReg 0.952 / 0.955, SVM 0.959 / 0.962


if __name__ == "__main__":
    main()
