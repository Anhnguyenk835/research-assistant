"""Unified instance for OpenSearch client."""

from functools import lru_cache
from typing import Optional

from config import Settings, get_settings

from .opensearch_client import OpenSearchClient


@lru_cache(maxsize=1)
def make_opensearch_client(settings: Settings = None) -> OpenSearchClient:
    """Factory function to create cached OpenSearch client.

    Uses lru_cache to maintain a singleton instance for efficiency.

    :param settings: Optional settings instance
    :returns: Cached OpenSearchClient instance
    """
    if settings is None:
        settings = get_settings()

    return OpenSearchClient(host=settings.opensearch.host, settings=settings)


def make_opensearch_client_fresh(settings: Settings = None) -> OpenSearchClient:
    """Factory function to create a fresh OpenSearch client (not cached).

    Use this when you need a new client instance (e.g., for testing
    or when connection issues occur).

    :param settings: Optional settings instance
    :returns: New OpenSearchClient instance
    """
    if settings is None:
        settings = get_settings()

    # Use provided settings host
    opensearch_host = settings.opensearch.host

    return OpenSearchClient(host=opensearch_host, settings=settings)