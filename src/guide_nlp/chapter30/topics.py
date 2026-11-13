"""Topic modelling with BERTopic on Solon embeddings (chapter 30).

Usage: uv run --group neural python -m guide_nlp.chapter30.topics
"""

from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from spacy.lang.fr.stop_words import STOP_WORDS
from umap import UMAP

from guide_nlp.chapter28.vectors import SOLON, corpus_vectors
from guide_nlp.corpus.load import load_articles

STOP = sorted(STOP_WORDS | {"plus", "ainsi", "également", "aussi", "été"})


def main() -> None:
    texts = [a.text for a in load_articles()]
    embeddings, _ = corpus_vectors(SOLON)
    topic_model = BERTopic(
        language="multilingual",  # never "english" here: it would delete the accents
        umap_model=UMAP(
            n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=1
        ),
        hdbscan_model=HDBSCAN(min_cluster_size=15, metric="euclidean", prediction_data=True),
        vectorizer_model=CountVectorizer(stop_words=STOP, token_pattern=r"[^\W\d_]{3,}"),
    )
    topics, _ = topic_model.fit_transform(texts, embeddings=embeddings)
    print(f"{max(topics) + 1} topics, {topics.count(-1)} outliers")
    print(topic_model.get_topic_info().head(10))


if __name__ == "__main__":
    main()
