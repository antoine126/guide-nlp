"""The measurements of chapter 15: every variant, errors, learning curve, k sweep.

Usage: uv run --group classic python -m guide_nlp.chapter15.experiments
"""

from collections import Counter

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    confusion_matrix,
    f1_score,
    normalized_mutual_info_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from spacy.lang.fr.stop_words import STOP_WORDS

from guide_nlp.chapter15.baselines import TFIDF, baseline_models
from guide_nlp.chapter15.data import rubric_split, tokenize


def main() -> None:
    Xtr, ytr, Xte, yte = rubric_split()
    models = {
        "majority class": make_pipeline(DummyClassifier(strategy="most_frequent")),
        **baseline_models(),
        "SVM + uni+bigrams": make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), **TFIDF), LinearSVC()
        ),
        "SVM + char 3-5": make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), **TFIDF), LinearSVC()
        ),
        "SVM + no stop words": make_pipeline(
            TfidfVectorizer(stop_words=list(STOP_WORDS), **TFIDF), LinearSVC()
        ),
    }
    predictions = {}
    for name, model in models.items():
        predictions[name] = model.fit(Xtr, ytr).predict(Xte)
        print(
            f"{name:<20} accuracy={accuracy_score(yte, predictions[name]):.3f} "
            f"macro F1={f1_score(yte, predictions[name], average='macro'):.3f}"
        )

    labels = sorted(set(ytr))
    print(labels)
    print(confusion_matrix(yte, predictions["SVM + TF-IDF"], labels=labels))
    errors = sum(y != p for y, p in zip(yte, predictions["SVM + TF-IDF"], strict=True))
    print(f"{errors} errors out of {len(yte)} test articles")

    order = np.random.default_rng(0).permutation(len(Xtr))
    print("learning curve (train size: macro F1 of NB, LogReg, SVM)")
    for n in (12, 30, 60, 120, 300, 600, 1200, len(Xtr)):
        sub, y = [Xtr[i] for i in order[:n]], [ytr[i] for i in order[:n]]
        row = []
        classifiers: list[ClassifierMixin] = [
            MultinomialNB(),
            LogisticRegression(max_iter=2000),
            LinearSVC(),
        ]
        for classifier in classifiers:
            pipe = make_pipeline(TfidfVectorizer(sublinear_tf=True), classifier).fit(sub, y)
            row.append(round(f1_score(yte, pipe.predict(Xte), average="macro"), 3))
        print(f"  {n:>5}: {row}")

    tv = TfidfVectorizer(
        tokenizer=tokenize, token_pattern=None, sublinear_tf=True, min_df=5, max_df=0.5
    )
    X = tv.fit_transform(Xtr)
    for k in (2, 4, 6, 8, 10, 15, 20):
        km = KMeans(k, n_init=5, random_state=0).fit(X)
        print(
            f"k={k:<3} ARI={adjusted_rand_score(ytr, km.labels_):.3f} "
            f"NMI={normalized_mutual_info_score(ytr, km.labels_):.3f}"
        )
    km = KMeans(6, n_init=10, random_state=0).fit(X)
    print(Counter(km.labels_.tolist()))


if __name__ == "__main__":
    main()
