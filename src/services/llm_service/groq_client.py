import logging
from groq import Groq
from typing import Dict, Any, List, Optional

from config import Settings
from .prompt import GENERAL_SYSTEM_PROMPT

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
            # Build context from search results
            context_parts = []
            sources = []

            for i, result in enumerate(search_results, 1):
                chunk_data = result.get('chunk', {})
                arxiv_meta = chunk_data.get('arxiv_metadata', {})
                chunk_meta = chunk_data.get('chunk_metadata', {})
                
                # Extract relevant information
                arxiv_id = arxiv_meta.get('arxiv_id', 'N/A')
                title = arxiv_meta.get('title', 'N/A')
                section = chunk_meta.get('section_heading', 'N/A')
                text = chunk_data.get('chunk_text', '')
                score = result.get('score', 0)
                prov = chunk_meta.get('prov', None)

                # Add to context
                context_parts.append(
                    f"[Source {i}] (Score: {score:.3f})\n"
                    f"Paper: {title} (arXiv:{arxiv_id})\n"
                    f"Section: {section}\n"
                    f"Content: {text}\n"
                )

                # Track source metadata
                source_dict = {
                    "rank": i,
                    "arxiv_id": arxiv_id,
                    "title": title,
                    "section": section,
                    "score": score,
                    "chunk_id": result.get('chunk_id', 'N/A'),
                    "chunk_text": text
                }
                
                # Add prov if available
                if prov:
                    source_dict["prov"] = prov
                
                sources.append(source_dict)

            # Combine context
            context = "\n---\n".join(context_parts)

            # Default system prompt for RAG
            default_system_prompt = (
                "You are a helpful research assistant. Answer the user's question based on the provided research paper excerpts. "
                "Be precise and cite the sources using [Source N] notation. If the provided context doesn't contain enough information, "
                "acknowledge what you don't know rather than making assumptions."
            )

            # Build messages
            messages = [
                {
                    "role": "system",
                    "content": system_prompt or default_system_prompt
                },
                {
                    "role": "user",
                    "content": f"Context from research papers:\n\n{context}\n\nQuestion: {query}\n\nAnswer:"
                }
            ]

            # Use Groq SDK to generate completion
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
                "sources": sources,
                "query": query,
                "num_sources": len(sources),
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
