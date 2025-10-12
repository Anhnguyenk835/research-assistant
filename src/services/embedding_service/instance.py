from typing import Optional
from dotenv import load_dotenv
from config import Settings, get_settings

from .jina_client import JinaEmbeddingsClient

load_dotenv()

def make_embeddings_client(settings: Optional[Settings] = None) -> JinaEmbeddingsClient:
    """Factory function to create embeddings service.

    Creates a new client instance each time to avoid closed client issues.

    :param settings: Optional settings instance
    :returns: JinaEmbeddingsClient instance
    """
    if settings is None:
        settings = get_settings()

    # Get API key from settings
    api_key = settings.jina.api_key if settings else os.getenv("JINA_API_KEY")

    return JinaEmbeddingsClient(api_key=api_key)
