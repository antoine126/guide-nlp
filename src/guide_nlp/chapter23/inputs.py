"""What a pretrained tokenizer gives a model: ids, segments, masks, offsets (chapter 23).

Usage: uv run --group neural python -m guide_nlp.chapter23.inputs
"""

from transformers import AutoTokenizer


def bert_inputs() -> None:
    bert = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")
    enc = bert("Le chat dort.", "Il rêve.")
    print(bert.convert_ids_to_tokens(enc["input_ids"]))
    # ['[CLS]', 'Le', 'chat', 'dort', '.', '[SEP]', 'Il', 'rêve', '.', '[SEP]']
    print(enc["token_type_ids"])  # [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]


def camembert_inputs() -> None:
    camembert = AutoTokenizer.from_pretrained("almanach/camembert-base")
    batch = camembert(
        ["Bonjour", "Le chat dort sur le canapé."], padding=True, truncation=True, max_length=8
    )
    print(batch["input_ids"])  # [[5, 1285, 6, 1, 1, 1, 1, 1],
    #  [5, 54, 1234, 16718, 32, 16, 4910, 6]]
    print(batch["attention_mask"])  # [[1, 1, 1, 0, 0, 0, 0, 0],
    #  [1, 1, 1, 1, 1, 1, 1, 1]]

    enc = camembert("L'homme d'aujourd'hui", return_offsets_mapping=True)
    print(camembert.convert_ids_to_tokens(enc["input_ids"]))
    # ['<s>', '▁L', "'", 'homme', '▁d', "'", 'aujourd', "'", 'hui', '</s>']
    print(enc["offset_mapping"])
    # [(0, 0), (0, 1), (1, 2), (2, 7), (8, 9), (9, 10), (10, 17), (17, 18), (18, 21), (0, 0)]
    print(enc.word_ids())
    # [None, 0, 0, 0, 1, 1, 1, 1, 1, None]


def gpt2_quirks() -> None:
    gpt2 = AutoTokenizer.from_pretrained("openai-community/gpt2")
    print(gpt2.encode("chat"), gpt2.encode(" chat"))  # [17006] [8537]
    print(gpt2.tokenize(" Paris"))  # ['ĠParis']
    print(gpt2.tokenize(" PARIS"))  # ['ĠPAR', 'IS']
    print(gpt2.tokenize(" paris"))  # ['Ġpar', 'is']
    print(gpt2.tokenize("1234567"))  # ['123', '45', '67']
    print(gpt2.encode(" SolidGoldMagikarp"))  # [43453]


def main() -> None:
    bert_inputs()
    camembert_inputs()
    gpt2_quirks()


if __name__ == "__main__":
    main()
