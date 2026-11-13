"""A classification service with FastAPI (chapter 46).

Usage: uv run --group serve --group neural uvicorn guide_nlp.chapter46.service:app
then: curl -X POST localhost:8000/rubrique -H 'content-type: application/json' \\
           -d '{"text": "Le télescope a observé une nouvelle galaxie."}'
The classifier is a logistic regression on Solon embeddings, trained at start-up.
"""

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression

from guide_nlp.chapter28.vectors import SOLON, corpus_vectors, load
from guide_nlp.corpus.load import load_articles

state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(application: FastAPI):
    articles = load_articles()
    vectors, _ = corpus_vectors(SOLON)
    train = [i for i, a in enumerate(articles) if a.split == "train"]
    state["classifier"] = LogisticRegression(max_iter=3000).fit(
        vectors[train], [articles[i].rubric for i in train]
    )
    state["encoder"] = load(SOLON, max_seq_length=512)
    yield
    state.clear()


app = FastAPI(lifespan=lifespan)


class Query(BaseModel):
    text: str


@app.post("/rubrique")
def classify(query: Query) -> dict:
    encoder, classifier = state["encoder"], state["classifier"]
    vector = encoder.encode([query.text], normalize_embeddings=True)
    return {"rubrique": str(classifier.predict(vector)[0])}
