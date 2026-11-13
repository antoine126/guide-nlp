"""Condensing a follow-up question into a standalone one (chapter 37).

Usage: uv run --group llm python -m guide_nlp.chapter37.condense
"""

from guide_nlp.rag import llm_text


def llm(prompt: str) -> str:
    return llm_text(prompt)


def condense(history: list[tuple[str, str]], question: str) -> str:
    lines = "\n".join(f"{role} : {text}" for role, text in history)
    return llm(
        f"Voici une conversation.\n{lines}\n\nDernière question de l'utilisateur : "
        f"{question}\n\nRéécris cette dernière question pour qu'elle se comprenne "
        f"seule, sans la conversation. Réponds uniquement par la question."
    )


if __name__ == "__main__":
    history = [
        ("utilisateur", "Qui a présenté le budget du Québec de novembre 2001 ?"),
        ("assistant", "Pauline Marois. [1]"),
    ]
    print(condense(history, "Et quel poste occupait-elle alors ?"))
