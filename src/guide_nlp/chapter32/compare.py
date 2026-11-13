"""Three local models behind the same OpenAI-compatible endpoint (chapter 32).

Usage: uv run --group llm python -m guide_nlp.chapter32.compare
(with Ollama running and the three models pulled)
"""

import httpx

from guide_nlp.config import settings

BASE_URL = f"{settings.ollama_url}/v1"


def ask(model: str, prompt: str) -> dict:
    response = httpx.post(
        f"{BASE_URL}/chat/completions",
        json={"model": model, "temperature": 0, "messages": [{"role": "user", "content": prompt}]},
        timeout=600,
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    for model in ("smollm2:135m", "qwen2.5:7b-instruct-q4_K_M", "deepseek-r1:8b"):
        answer = ask(model, "Quelle est la capitale de l'Australie ?")
        print(model, answer["usage"], answer["choices"][0]["message"]["content"][:80])


if __name__ == "__main__":
    main()
