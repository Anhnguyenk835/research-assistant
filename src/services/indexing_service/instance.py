from typing import Optional

from config import Settings, get_settings
from services.embedding_service.instance import make_embeddings_client
from services.opensearch_service.instance import make_opensearch_client_fresh

from .indexer import HybridIndexingService
from .chunker import HeadingChunk


def make_hybrid_indexing_service(settings: Settings = None) -> HybridIndexingService:
    """Factory function to create hybrid indexing service.

    Creates a new service instance each time.

    :param settings: Optional settings instance
    :returns: HybridIndexingService instance
    """
    if settings is None:
        settings = get_settings()

    # Create dependencies using configuration
    chunker = HeadingChunk(
        max_chunk_size=settings.chunking.max_chunk_size,
        min_chunk_size=settings.chunking.min_chunk_size,
        overlap=settings.chunking.overlap_size,
    )
    embeddings_client = make_embeddings_client(settings)
    opensearch_client = make_opensearch_client_fresh(settings)

    # Create indexing service
    return HybridIndexingService(chunker=chunker, embedding_client=embeddings_client, opensearch_client=opensearch_client)