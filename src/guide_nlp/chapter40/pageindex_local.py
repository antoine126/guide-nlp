"""PageIndex with a local model through LiteLLM (chapter 40).

PageIndex pins an older huggingface-hub: it has its own environment.
Usage:
    uv sync --project envs/pageindex
    OLLAMA_API_BASE=http://localhost:11434 \\
        uv run --project envs/pageindex python -m guide_nlp.chapter40.pageindex_local
"""

from pageindex import PageIndexClient

from guide_nlp.config import settings

MODEL = "ollama_chat/qwen3:8b"


def main() -> None:
    client = PageIndexClient(
        index_model=MODEL, chat_model=MODEL, storage_path=str(settings.interim_dir / "pageindex")
    )
    pdf = settings.raw_dir / "rapports" / "anssi_panorama_2024.pdf"
    document = client.submit_document(str(pdf), wait=True)
    print(document["usage"])
    print(
        client.chat(
            "Quel type d'attaques a le plus mobilisé l'ANSSI en 2024 ?", doc_id=document["doc_id"]
        )
    )


if __name__ == "__main__":
    main()
