"""The same system with LlamaIndex (chapter 35).

Usage: uv run --group rag python -m guide_nlp.chapter35.llama_rag
Indexing the 2,923 articles takes a few minutes the first time.
"""

from llama_index.core import Document, Settings, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

from guide_nlp.config import settings
from guide_nlp.corpus.load import load_articles


def main() -> None:
    articles = load_articles()
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="OrdalieTech/Solon-embeddings-large-0.1",
        query_instruction="query : ",
        embed_batch_size=64,
    )
    Settings.llm = Ollama(
        model=settings.llm_model, base_url=settings.ollama_url, request_timeout=600, temperature=0
    )
    Settings.text_splitter = SentenceSplitter(chunk_size=256, chunk_overlap=40)

    documents = [Document(text=a.text, metadata={"title": a.title}) for a in articles]
    engine = VectorStoreIndex.from_documents(documents).as_query_engine(similarity_top_k=5)
    response = engine.query("Qui a fondé cette chambre de commerce ?")
    print(response, [node.metadata["title"] for node in response.source_nodes])


if __name__ == "__main__":
    main()
