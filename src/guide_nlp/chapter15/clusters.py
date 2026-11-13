"""K-means on TF-IDF vectors, compared with the six rubrics (chapter 15).

Usage: uv run --group classic python -m guide_nlp.chapter15.clusters
"""

from collections import Counter

import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from guide_nlp.chapter15.data import rubric_split, tokenize


def main() -> None:
    train_texts, train_labels, _, _ = rubric_split()
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize,
        token_pattern=None,  # no stop words
        sublinear_tf=True,
        min_df=5,
        max_df=0.5,
    )
    X = vectorizer.fit_transform(train_texts)
    terms = vectorizer.get_feature_names_out()

    kmeans = KMeans(6, n_init=10, random_state=0).fit(X)
    for c, center in enumerate(kmeans.cluster_centers_):
        members = Counter(
            y for y, label in zip(train_labels, kmeans.labels_, strict=True) if label == c
        )
        print(c, [terms[i] for i in np.argsort(-center)[:8]], members.most_common(2))
    print(
        f"ARI={adjusted_rand_score(train_labels, kmeans.labels_):.3f} "
        f"NMI={normalized_mutual_info_score(train_labels, kmeans.labels_):.3f}"
    )
    # Book: ARI=0.704 NMI=0.719


if __name__ == "__main__":
    main()
