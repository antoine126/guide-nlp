"""Calling a model through an API: the Anthropic SDK (chapter 32).

Usage: ANTHROPIC_API_KEY=... uv run --group llm python -m guide_nlp.chapter32.claude_api
The book measures only local models: no API model is evaluated in it.
"""

import anthropic


def main() -> None:
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    response = client.messages.create(
        model="claude-opus-5-5",
        max_tokens=16000,
        messages=[{"role": "user", "content": "Résume ce rapport en trois points : ..."}],
    )
    print(next(block.text for block in response.content if block.type == "text"))
    print(response.usage.input_tokens, response.usage.output_tokens)


if __name__ == "__main__":
    main()
