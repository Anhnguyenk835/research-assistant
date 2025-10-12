import os
from pathlib import Path
from typing import Optional, List, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

load_dotenv()

class BaseConfigSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8", 
        extra="ignore",
        frozen=True,
        env_nested_delimiter="__",
        case_sensitive=False
    )

class ArxivSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_prefix="ARXIV_",
        frozen=True,
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    base_url: str = Field(default="https://export.arxiv.org/api/query")
    rate_limit_delay: float = Field(default=3.0, description="Delay in seconds between API requests to respect rate limits.")
    timeout_seconds: int = Field(default=30, description="Timeout for API requests in seconds.")
    max_results: int = Field(default=1, description="Maximum number of results to fetch per query.")
    search_category: str = Field(default="cs.AI", description="Default category to search in arXiv.")
    download_max_retries: int = Field(default=3, description="Number of retries for downloading PDFs.")
    download_retry_delay: float = Field(default=5.0, description="Delay in seconds between download retries.")
    max_concurrent_downloads: int = Field(default=5, description="Maximum number of concurrent PDF downloads.")
    max_concurrent_parsing: int = Field(default=3, description="Maximum number of concurrent PDF parsing operations.")
    pdf_cache_dir: str = ".data/arxiv_pdfs"

    namespace: dict = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
        "opensearch": "http://a9.com/-/spec/opensearch/1.1/"
    }

    @field_validator("pdf_cache_dir")
    @classmethod
    def validate_pdf_cache_dir(cls, value: str) -> str:
        os.makedirs(value, exist_ok=True)
        return value
    

class PDFParserSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_prefix="PARSER_",
        frozen=True,
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    max_pages: int = Field(default=25, description="Maximum number of pages to parse from a PDF.")
    max_file_size_mb: int = Field(default=20, description="Maximum file size in megabytes for a PDF to be parsed.")
    ocr_option: bool = Field(default=True, description="Whether to enable OCR for scanned PDFs.")
    table_extraction: bool = Field(default=True, description="Whether to enable table extraction from PDFs.")


class ChunkingSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_prefix="CHUNKING_",
        frozen=True,
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    max_chunk_size: int = Field(default=500, description="Maximum number of words in a text chunk.")
    min_chunk_size: int = Field(default=100, description="Minimum number of words in a text chunk.")
    overlap_size: int = Field(default=0, description="Number of overlapping words between consecutive chunks.")


class OpenSearchSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_prefix="OPENSEARCH_",
        frozen=True,
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    host: str = os.getenv("OPENSEARCH_HOST", "http://localhost:9200")
    username: Optional[str] = os.getenv("OPENSEARCH_USERNAME", "admin")
    password: Optional[str] = os.getenv("OPENSEARCH_PASSWORD", "Tanh080305@")

    index_name: str = "arxiv-papers"
    chunk_index_suffix: str = "chunks"  # Creates single hybrid index: {index_name}-{suffix}
    max_text_size: int = 1000000

    # Vector search settings
    vector_dimension: int = 1024  # Jina embeddings dimension
    vector_space_type: str = "cosinesimil"  # cosinesimil, l2, innerproduct

    # Hybrid search settings
    rrf_pipeline_name: str = "hybrid-rrf-pipeline"
    hybrid_search_size_multiplier: int = 2  # Get k*multiplier for better recall


class GroqSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_prefix="GROQ_",
        frozen=True,
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    api_key: str = Field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    model: str = Field(default="llama-3.3-70b-versatile", description="Groq model to use for LLM")
    base_url: str = Field(default="https://api.groq.com/openai/v1", description="Groq API base URL")
    temperature: float = Field(default=0.3, description="Temperature for RAG responses (lower = more factual)")
    max_tokens: int = Field(default=2048, description="Maximum tokens for response generation")
    timeout_seconds: int = Field(default=60, description="Timeout for API requests in seconds")

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        if not v:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        return v


class JinaSettings(BaseConfigSettings):
    model_config = SettingsConfigDict(
        env_prefix="JINA_",
        frozen=True,
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    api_key: str = Field(default_factory=lambda: os.getenv("JINA_API_KEY", ""))
    base_url: str = Field(default="https://api.jina.ai/v1", description="Jina API base URL")
    model: str = Field(default="jina-embeddings-v3", description="Jina embedding model")
    dimension: int = Field(default=1024, description="Embedding dimension")
    timeout_seconds: int = Field(default=30, description="Timeout for API requests in seconds")

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        if not v:
            raise ValueError("JINA_API_KEY environment variable is not set")
        return v


class Settings(BaseConfigSettings):
    app_version: str = "0.1.0"
    debug: bool = True
    environment: Literal["development", "staging", "production"] = "development"
    service_name: str = "rag-api"

    arxiv: ArxivSettings = Field(default_factory=ArxivSettings)
    pdf_parser: PDFParserSettings = Field(default_factory=PDFParserSettings)
    opensearch: OpenSearchSettings = Field(default_factory=OpenSearchSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    groq: GroqSettings = Field(default_factory=GroqSettings)
    jina: JinaSettings = Field(default_factory=JinaSettings)


def get_settings() -> Settings:
    return Settings()
