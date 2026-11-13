"""Chat templates, and the size of the KV cache (chapter 31).

Usage: uv run --group neural python -m guide_nlp.chapter31.template
"""

from transformers import AutoConfig, AutoTokenizer

NAME = "Qwen/Qwen2.5-7B-Instruct"


def kv_cache_bytes(name: str, tokens: int, bytes_per_value: int = 2) -> float:
    """Keys and values of every layer, for every KV head, in fp16 by default."""
    config = AutoConfig.from_pretrained(name)
    head_dim = config.hidden_size // config.num_attention_heads
    layers, kv_heads = config.num_hidden_layers, config.num_key_value_heads
    return 2 * layers * kv_heads * head_dim * tokens * bytes_per_value


def main() -> None:
    tok = AutoTokenizer.from_pretrained(NAME)
    messages = [{"role": "user", "content": "Quelle est la capitale de la France ?"}]
    print(tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    config = AutoConfig.from_pretrained(NAME)
    head_dim = config.hidden_size // config.num_attention_heads
    print(config.num_hidden_layers, config.num_key_value_heads, head_dim)  # 28 4 128
    print(f"{kv_cache_bytes(NAME, 2048) / 1e6:.2f} MB")  # 117.44 MB


if __name__ == "__main__":
    main()
