"""Text-to-text with T5: the task is written in the input (chapter 26).

Usage: uv run --group neural python -m guide_nlp.chapter26.t5
"""

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("google-t5/t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("google-t5/t5-small").eval()


def generate(text: str) -> str:
    enc = tok(text, return_tensors="pt", truncation=True)
    return tok.decode(model.generate(**enc, max_new_tokens=60)[0], skip_special_tokens=True)


if __name__ == "__main__":
    print(generate("translate English to French: The house is wonderful."))
    # La maison est merveilleuse.
    print(generate("The house is wonderful."))
    # Das Haus ist wunderbar.
