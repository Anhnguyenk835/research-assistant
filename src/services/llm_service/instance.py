"""Factory function for creating Groq LLM client instance."""

from dotenv import load_dotenv
from services.llm_service.groq_client import GroqClient
from config import Settings

load_dotenv() 

def make_groq_client(settings: Settings = None, model: str = None) -> GroqClient:
    """Create and return a Groq client instance.
    
    :param settings: Optional Settings object
    :param model: Optional model name override
    :returns: Configured GroqClient instance
    """
    # Get API key from environment
    api_key = settings.groq.api_key if settings else os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set")
    
    # Get model from environment or use default
    default_model = settings.groq.model if settings else "llama-3.3-70b-versatile"
    model_to_use = model or default_model
    
    return GroqClient(api_key=api_key, model=model_to_use)
