"""Fine-tuning a bi-encoder on (title, beginning of the article) pairs (chapter 27).

Usage: uv run --group neural python -m guide_nlp.chapter27.finetune
The pairs need no annotation: the in-batch negatives of MultipleNegativesRankingLoss
do the rest. The script reports nDCG@10 on the 112 queries before and after.
"""

from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)

from guide_nlp.chapter13.metrics import evaluate_scores, mean_metrics
from guide_nlp.config import settings
from guide_nlp.corpus.load import load_articles, load_queries

NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def ndcg(model: SentenceTransformer) -> float:
    articles, queries = load_articles(), load_queries()
    docs = model.encode([a.text for a in articles], batch_size=64, normalize_embeddings=True)
    qs = model.encode([q.query for q in queries], normalize_embeddings=True)
    rows = evaluate_scores(
        qs @ docs.T, [a.pageid for a in articles], [set(q.relevant) for q in queries]
    )
    return mean_metrics(rows)["ndcg"]


def main() -> None:
    train = load_articles("train")
    model = SentenceTransformer(NAME)
    model.max_seq_length = 256
    print(f"before: nDCG@10 = {ndcg(model):.3f}")
    pairs = Dataset.from_dict(
        {
            "anchor": [a.title for a in train],
            "positive": [" ".join(a.text.split()[:150]) for a in train],
        }
    )
    args = SentenceTransformerTrainingArguments(
        output_dir=str(settings.interim_dir / "ch27_finetune"),
        num_train_epochs=1,
        per_device_train_batch_size=32,
        learning_rate=2e-5,
        warmup_steps=20,
        report_to="none",
        save_strategy="no",
        seed=0,
    )
    loss = losses.MultipleNegativesRankingLoss(model, scale=20.0)
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=pairs, loss=loss)
    trainer.train()
    print(f"after:  nDCG@10 = {ndcg(model):.3f}")


if __name__ == "__main__":
    main()
