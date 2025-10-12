"""FastAPI dependencies for dependency injection."""

from typing import AsyncGenerator
from functools import lru_cache
from fastapi import Request

from config import Settings, get_settings
from services.opensearch_service.opensearch_client import OpenSearchClient
from services.llm_service.groq_client import GroqClient
from services.embedding_service.jina_client import JinaEmbeddingsClient
from services.opensearch_service.instance import make_opensearch_client_fresh
from services.embedding_service.instance import make_embeddings_client
from services.llm_service.instance import make_groq_client


# Settings dependency
@lru_cache()
def get_settings_dependency() -> Settings:
    """Get application settings (cached)."""
    return get_settings()


# OpenSearch client dependency
async def get_opensearch_client() -> AsyncGenerator[OpenSearchClient, None]:
    """Get OpenSearch client instance.
    
    This is a dependency that yields the client and ensures cleanup.
    """
    settings = get_settings_dependency()
    client = make_opensearch_client_fresh(settings=settings)
    try:
        yield client
    finally:
        # Cleanup if needed (OpenSearchClient doesn't need explicit cleanup currently)
        pass


# Embeddings client dependency
async def get_embeddings_client(request: Request) -> AsyncGenerator[JinaEmbeddingsClient, None]:
    """Get Jina embeddings client instance.
    
    This is a dependency that yields the client and ensures cleanup.
    """
    return request.app.state.clients["embeddings_client"]


# Groq LLM client dependency
async def get_groq_client(request: Request) -> AsyncGenerator[GroqClient, None]:
    """Get Groq LLM client instance.
    
    This is a dependency that yields the client and ensures cleanup.
    """
    return request.app.state.clients["groq_client"]