"""Era 4: a pretrained model, used without any training example (chapter 1).

Usage: uv run --group neural python -m guide_nlp.chapter01.zero_shot
"""

from transformers import pipeline

from guide_nlp.chapter01.messages import MESSAGES

MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
LABELS = ["réclamation", "satisfaction", "question"]


def main() -> None:
    # Multilingual model trained on natural language inference (NLI).
    classifier = pipeline("zero-shot-classification", model=MODEL)
    for message in MESSAGES:
        result = classifier(
            message,
            candidate_labels=LABELS,
            hypothesis_template="Ce message est une {}.",
        )
        best_label, best_score = result["labels"][0], result["scores"][0]
        print(f"{best_label:<13} {best_score:.2f} | {message}")


if __name__ == "__main__":
    main()
