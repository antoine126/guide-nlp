"""Masked language modelling: the 80/10/10 corruption, and a fill-mask query (chapter 25).

Usage: uv run --group neural python -m guide_nlp.chapter25.mlm
"""

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, PreTrainedTokenizerBase

NAME = "almanach/camembert-base"


def mask_tokens(
    input_ids: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
    special_mask: torch.Tensor,
    p: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = input_ids.clone()
    chosen = torch.bernoulli(torch.full(input_ids.shape, p)).bool() & ~special_mask
    labels[~chosen] = -100  # the loss ignores the other tokens
    input_ids = input_ids.clone()
    replace = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & chosen
    input_ids[replace] = tokenizer.mask_token_id
    randomize = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & chosen & ~replace
    input_ids[randomize] = torch.randint(len(tokenizer), input_ids.shape)[randomize]
    return input_ids, labels


def main() -> None:
    tok = AutoTokenizer.from_pretrained(NAME)
    model = AutoModelForMaskedLM.from_pretrained(NAME).eval()

    torch.manual_seed(0)
    enc = tok("Le chat dort sur le canapé du salon depuis ce matin.", return_tensors="pt")
    special = torch.tensor(
        tok.get_special_tokens_mask(enc["input_ids"][0].tolist(), already_has_special_tokens=True)
    ).bool()
    corrupted, labels = mask_tokens(enc["input_ids"], tok, special[None], p=0.3)
    print(tok.convert_ids_to_tokens(corrupted[0]))
    print((labels != -100).sum().item(), "tokens to predict")

    enc = tok("La Terre tourne autour du <mask>.", return_tensors="pt")
    position = (enc["input_ids"][0] == tok.mask_token_id).nonzero().item()
    with torch.no_grad():
        top = model(**enc).logits[0, position].softmax(-1).topk(5)
    pairs = zip(top.values.tolist(), top.indices.tolist(), strict=True)
    print([(tok.decode(i).strip(), round(p, 3)) for p, i in pairs])


if __name__ == "__main__":
    main()
