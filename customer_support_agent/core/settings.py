from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    app_name: str = "AI Copilot for Support Agents"

    # LLM Configuration
    openai_api_key: str = ""
    # GPT-5 nano is the fastest, lowest-cost GPT-5 model and supports the
    # function calls used by the support agent.
    openai_model: str = "gpt-5-nano"
    llm_temperature: float = 0.2

    # A low-cost, production embedding model used with OPENAI_API_KEY.
    openai_embedding_model: str = "text-embedding-3-small"

    # Workspace & Storage
    workspace_dir: Path = Path(__file__).resolve().parents[2]

    data_dir: Path = Path("data")
    db_path: Path = Path("data/support.db")
    # Keep OpenAI vectors separate from previous provider vector stores.
    chroma_rag_dir: Path = Path("data/chroma_rag_openai")
    chroma_mem0_dir: Path = Path("data/chroma_mem0_openai")
    knowledge_base_dir: Path = Path("knowledge_base")

    # RAG Configuration
    rag_chunk_size: int = 800
    rag_chunk_overlap: int = 120
    rag_top_k: int = 4
    mem0_top_k: int = 5

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    dashboard_api_url: str = "http://localhost:8000"

    def resolve(self, path: Path) -> Path:
        """Resolve relative paths against the project root."""
        return path if path.is_absolute() else self.workspace_dir / path

    @property
    def db_file(self) -> Path:
        return self.resolve(self.db_path)

    @property
    def chroma_rag_path(self) -> Path:
        return self.resolve(self.chroma_rag_dir)

    @property
    def chroma_mem0_path(self) -> Path:
        return self.resolve(self.chroma_mem0_dir)

    @property
    def knowledge_base_path(self) -> Path:
        return self.resolve(self.knowledge_base_dir)

@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.
    The Settings object is created only once during the application's lifetime.
    """
    return Settings()


def ensure_directories(settings: Settings | None = None) -> None:
    """
    Create the local directories required by SQLite and ChromaDB.
    """
    config = settings or get_settings()

    for path in (
        config.resolve(config.data_dir),
        config.chroma_rag_path,
        config.chroma_mem0_path,
        config.knowledge_base_path,
    ):
        path.mkdir(parents=True, exist_ok=True)
