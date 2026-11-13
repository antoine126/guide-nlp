"""Pinned model loading (chapter 2)."""

from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

MODEL_ID = "almanach/camembert-base"
REVISION = "main"  # replace by a commit hash for strict reproducibility


def load_encoder(
    model_id: str = MODEL_ID, revision: str = REVISION
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Load a tokenizer/model pair pinned to a given revision."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModel.from_pretrained(model_id, revision=revision)
    assert tokenizer is not None, f"no tokenizer for {model_id}"
    return tokenizer, model
