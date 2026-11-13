"""An agent that decides its own searches, with a tool-calling LLM (chapter 39).

Usage: uv run --group neural --group llm python -m guide_nlp.chapter39.agent
"""

from collections.abc import Callable

import httpx

from guide_nlp.config import settings
from guide_nlp.rag import RagBase

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "chercher",
            "description": "Cherche dans l'encyclopédie et renvoie les extraits les plus proches.",
            "parameters": {
                "type": "object",
                "required": ["requete"],
                "properties": {
                    "requete": {
                        "type": "string",
                        "description": "Ce qu'il faut chercher, avec les noms propres utiles.",
                    }
                },
            },
        },
    }
]
SYSTEM = (
    "Tu réponds aux questions à l'aide de l'outil chercher. Tu ne dois rien affirmer "
    "sans l'avoir lu dans les extraits. Si la question passe par un intermédiaire, "
    "cherche d'abord cet intermédiaire, puis fais une nouvelle recherche sur lui. "
    "Si tu ne trouves pas, réponds exactement : « Je ne sais pas. »"
)


def chat(messages: list[dict], tools: list | None) -> dict:
    payload = {
        "model": settings.llm_model,
        "stream": False,
        "messages": messages,
        "options": {"temperature": 0},
    }
    if tools:
        payload["tools"] = tools
    reply = httpx.post(f"{settings.ollama_url}/api/chat", json=payload, timeout=600)
    reply.raise_for_status()
    return reply.json()["message"]


def agent(question: str, search: Callable[[str], str], budget: int = 3) -> str:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    for step in range(budget + 1):
        # once the budget is spent, the model has no tool left and must answer
        message = chat(messages, TOOLS if step < budget else None)
        calls = message.get("tool_calls")
        if not calls:
            return message["content"]
        messages.append(message)
        for call in calls:
            query = call["function"]["arguments"].get("requete", question)
            print(f"  search: {query}")
            messages.append({"role": "tool", "tool_name": "chercher", "content": search(query)})
    return "Je ne sais pas."


def main() -> None:
    rag = RagBase()

    def search(query: str) -> str:
        return "\n\n".join(rag.chunks[i] for i in rag.dense(query, 5))

    for question in (
        "Dans quelle ville est mort le père de Bardellone Bonacossi ?",
        "Quel était le prénom du chien de Furcy Madeleine ?",
    ):
        print(question)
        print("->", agent(question, search))


if __name__ == "__main__":
    main()
