"""Word error rate, after normalising case and punctuation (chapter 45).

Usage: uv run --group audio python -m guide_nlp.chapter45.wer
"""

import jiwer

normalize = jiwer.Compose(
    [jiwer.ToLowerCase(), jiwer.RemovePunctuation(), jiwer.RemoveMultipleSpaces(), jiwer.Strip()]
)


def word_error_rate(reference: str, hypothesis: str) -> float:
    return jiwer.wer(normalize(reference), normalize(hypothesis))


if __name__ == "__main__":
    reference = "ces droits de plantation étaient intégrés dans la réforme de deux mille huit"
    hypothesis = "ces droits d'éplantation étaient intégrés dans la réforme 2008"
    print(f"WER = {word_error_rate(reference, hypothesis):.2f}")
