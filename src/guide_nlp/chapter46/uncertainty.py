"""Is model A really better than model B? Bootstrap and McNemar (chapter 46).

Usage: uv run --group classic python -m guide_nlp.chapter46.uncertainty
Compares the TF-IDF + linear SVM and the TF-IDF + logistic regression of chapter 15.
"""

import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import f1_score

from guide_nlp.chapter15.baselines import baseline_models
from guide_nlp.chapter15.data import rubric_split

rng = np.random.default_rng(0)


def bootstrap_difference(y: np.ndarray, a: np.ndarray, b: np.ndarray, n: int = 2000) -> np.ndarray:
    values = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        values.append(f1_score(y[i], a[i], average="macro") - f1_score(y[i], b[i], average="macro"))
    return np.percentile(values, [2.5, 97.5])


def mcnemar(y: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    only_a = int(np.sum((a == y) & (b != y)))
    only_b = int(np.sum((a != y) & (b == y)))
    return binomtest(min(only_a, only_b), only_a + only_b, 0.5).pvalue


def main() -> None:
    train_texts, train_labels, test_texts, test_labels = rubric_split()
    models = baseline_models()
    y = np.array(test_labels)
    a = models["SVM + TF-IDF"].fit(train_texts, train_labels).predict(test_texts)
    b = models["LogReg + TF-IDF"].fit(train_texts, train_labels).predict(test_texts)
    low, high = bootstrap_difference(y, a, b)
    print(
        f"macro F1: SVM {f1_score(y, a, average='macro'):.3f}, "
        f"LogReg {f1_score(y, b, average='macro'):.3f}"
    )
    print(
        f"95 % interval of the difference: [{low:+.3f}, {high:+.3f}]; "
        f"McNemar p = {mcnemar(y, a, b):.2f}"
    )


if __name__ == "__main__":
    main()
