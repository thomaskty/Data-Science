import os
from pathlib import Path
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """
    Configuration settings for the multi-modal RAG system
    """

    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    SAMPLE_DATA_DIR: Path = PROJECT_ROOT / "sample_data"
    MEDIA_STORAGE_DIR: Path = PROJECT_ROOT / "media_storage"

    # Database Configuration
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int

    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    # OpenAI Configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Model Configuration
    EMBEDDING_MODEL: str = "text-embedding-3-small"  # $0.02 per 1M tokens (cheapest)
    EMBEDDING_DIMENSIONS: int = 1536  # for text-embedding-3-small
    CHAT_MODEL: str = "gpt-4o-mini"  # $0.15/$0.60 per 1M tokens (much cheaper than gpt-4o)
    VISION_MODEL: str = "gpt-4o-mini"  # $0.15/$0.60 per 1M tokens (supports vision, 85% cheaper)

    # Chunking Configuration
    CHUNK_SIZE: int = 500  # tokens
    CHUNK_OVERLAP: int = 50  # tokens

    # Retrieval Configuration
    TOP_K_RETRIEVAL: int = 20  # Initial retrieval
    TOP_K_RERANK: int = 5  # After reranking
    HYBRID_ALPHA: float = 0.5  # 0=BM25 only, 1=Vector only, 0.5=balanced

    # Reranking Configuration
    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    USE_RERANKING: bool = True

    # Image Processing
    MAX_IMAGE_SIZE: tuple = (1024, 1024)  # Max dimensions
    IMAGE_QUALITY: int = 85  # JPEG quality
    EXTRACT_TABLES_AS_IMAGES: bool = True

    # Graph Configuration
    BUILD_GRAPH: bool = True
    GRAPH_STORAGE_PATH: Path = PROJECT_ROOT / "graph_data" / "graph.gpickle"

    class Config:
        env_file = ".env"
        case_sensitive = True


# Create global settings instance
settings = Settings()

# Create necessary directories
settings.MEDIA_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
settings.GRAPH_STORAGE_PATH.parent.mkdir(parents=True, exist_ok=True)
settings.SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)