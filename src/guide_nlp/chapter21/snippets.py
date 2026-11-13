"""BLEU with several references, computed by sacrebleu (chapter 21).

Usage: uv run --group neural python -m guide_nlp.chapter21.snippets
"""

import sacrebleu


def main() -> None:
    hypotheses = ["he drank straight from the bottle .", "the reason is under obvious ."]
    references_1 = ["he drank straight from the bottle .", "the reason for this is obvious ."]
    references_2 = ["he drank right out of the bottle .", "the reason is obvious ."]
    references = [references_1, references_2]
    score = sacrebleu.corpus_bleu(hypotheses, references, tokenize="none").score
    print(f"BLEU = {score:.1f}")


if __name__ == "__main__":
    main()
