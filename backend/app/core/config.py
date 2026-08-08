"""
App configuration — all values come from environment variables (or .env file).
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Ollama
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"

    # ChromaDB vector store
    VECTOR_STORE_PATH: str = "data/vector_store"
    COLLECTION_NAME: str = "rag_docs"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Retrieval
    TOP_K: int = 4          # number of chunks to retrieve per query

    # CORS — comma-separated list of allowed origins
    CORS_ORIGINS: list[str] = ["http://localhost:8501", "http://localhost:7860", "*"]

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
