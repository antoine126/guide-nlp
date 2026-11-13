"""Project-wide settings (chapter 2)."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Project-wide settings (override with GUIDE_NLP_* env vars)."""

    model_config = SettingsConfigDict(env_prefix="GUIDE_NLP_", env_file=".env")

    data_dir: Path = Path("data")
    spacy_model: str = "fr_core_news_md"
    device: str = Field(default="auto", pattern="^(auto|cpu|cuda|mps)$")
    seed: int = 42
    ollama_url: str = "http://127.0.0.1:11434"
    llm_model: str = "qwen2.5:7b-instruct-q4_K_M"
    # The Wikimedia API asks every client to identify itself: put your contact here.
    user_agent: str = "guide-nlp/0.1 (companion code of 'Le Guide du NLP')"

    @property
    def raw_dir(self) -> Path:
        """Raw data, never modified."""
        return self.data_dir / "raw"

    @property
    def interim_dir(self) -> Path:
        """Intermediate results."""
        return self.data_dir / "interim"

    @property
    def processed_dir(self) -> Path:
        """Data ready for the chapters (the 'Articles' corpus, queries...)."""
        return self.data_dir / "processed"


settings = Settings()
