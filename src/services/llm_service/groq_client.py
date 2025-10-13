import logging
from groq import Groq
from typing import Dict, Any, List, Optional

from config import Settings
from .prompt import GENERAL_SYSTEM_PROMPT, build_rag_prompt, format_search_results_context
from schemas.rag.rag_models import RAGResponse

logger = logging.getLogger(__name__)


class GroqClient:
    """Client for Groq AI LLM API for chat completions and RAG."""

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        """Initialize Groq client.

        :param api_key: Groq API key
        :param model: Model name to use (default: llama-3.1-70b-versatile)
        :param base_url: API base URL
        """
        self.api_key = api_key
        self.model = model
        self.client = Groq(api_key=api_key)
        logger.info(f"Groq client initialized with model: {model}")

    async def health_check(self) -> bool:
        """Check if Groq API is accessible.

        :returns: True if API is healthy, False otherwise
        """
        try:
            # Simple API call to verify connectivity using Groq SDK
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_completion_tokens=10
            )
            logger.info("Groq API health check passed")
            return True
        except Exception as e:
            logger.error(f"Groq API health check failed: {e}")
            return False

    async def generate_answer(
        self,
        query: str,
        temperature: float = 0.4,
        max_tokens: int = 1024
    ) -> Dict[str, Any]:
        """Generate an answer using Groq LLM.

        :param query: User query
        :param system_prompt: Optional system prompt to guide the model
        :param temperature: Sampling temperature (0.0 to 2.0)
        :param max_tokens: Maximum tokens to generate
        :returns: Dictionary with answer and metadata
        """
        try:
            completion = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": GENERAL_SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                temperature=temperature,
                max_completion_tokens=max_tokens,
                top_p=1,
                stream=False,
                stop=None
            )

            # Extract answer from completion result (using dot notation for Groq SDK)
            answer = completion.choices[0].message.content
            
            logger.info(f"Generated answer for query: '{query[:50]}...'")

            return {
                "answer": answer,
                "model": completion.model,
                "usage": completion.usage.model_dump() if completion.usage else {},
                "finish_reason": completion.choices[0].finish_reason or "unknown"
            }

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise

    async def generate_rag_answer(
        self,
        query: str,
        search_results: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2048
    ) -> Dict[str, Any]:
        """Generate a RAG (Retrieval-Augmented Generation) answer using search results.

        :param query: User query
        :param search_results: List of search results from OpenSearch
        :param system_prompt: Optional system prompt
        :param temperature: Sampling temperature (lower for more factual)
        :param max_tokens: Maximum tokens to generate
        :returns: Dictionary with answer, sources, and metadata
        """
        try:

            # Save search results to json for debugging
            # import json
            # with open("search_results.json", "w") as f:
            #     json.dump(search_results, f, indent=2)

            # Build context from search results using the prompt helper function
            context = format_search_results_context(search_results, max_chunks=len(search_results))

            # Build RAG prompt using the prompt template with JSON schema
            system_prompt = build_rag_prompt(query, context, response_model=RAGResponse)

            # save system prompt to txt
            with open("system_prompt.txt", "w") as f:
                f.write(system_prompt)

            # Build messages with system prompt
            messages = [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": query
                }
            ]

            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=max_tokens
            )

            # Extract answer using dot notation (Groq SDK objects)
            answer = completion.choices[0].message.content

            logger.info(f"Generated RAG answer for query: '{query[:50]}...' using {len(search_results)} sources")

            return {
                "answer": answer,
                "query": query,
                "num_sources": len(search_results),
                "model": completion.model,
                "usage": completion.usage.model_dump() if completion.usage else {},
                "finish_reason": completion.choices[0].finish_reason or "unknown"
            }

        except Exception as e:
            logger.error(f"Error generating RAG answer: {e}")
            raise

    async def close(self):
        """Close the Groq client (synchronous close, wrapped in async)."""
        try:
            self.client.close()
            logger.info("Groq client closed")
        except Exception as e:
            logger.error(f"Error closing Groq client: {e}")

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
