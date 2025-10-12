"""RAG (Retrieval-Augmented Generation) API endpoints."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from dependencies import (
    get_opensearch_client,
    get_embeddings_client,
    get_groq_client
)
from services.opensearch_service.opensearch_client import OpenSearchClient
from services.embedding_service.jina_client import JinaEmbeddingsClient
from services.llm_service.groq_client import GroqClient
from schemas.rag import (
    RAGQueryRequest,
    RAGQueryResponse,
    SearchResultSource,
    AskRequest,
    AskResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/rag",
    tags=["RAG"],
    responses={
        404: {"description": "Not found"},
        500: {"description": "Internal server error"}
    }
)


@router.post("/query", response_model=RAGQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a question using RAG",
    description=(
        "Query the research paper database using hybrid search (BM25 + KNN vector search) "
        "and generate an answer using LLM with source citations. The answer will include "
        "references to specific papers and sections using [Source N] notation."
    )
)
async def query_rag(
    request: RAGQueryRequest,
    opensearch_client: Annotated[OpenSearchClient, Depends(get_opensearch_client)],
    embeddings_client: Annotated[JinaEmbeddingsClient, Depends(get_embeddings_client)],
    groq_client: Annotated[GroqClient, Depends(get_groq_client)]
) -> RAGQueryResponse:
    """Query research papers and generate an answer using RAG.
    
    This endpoint performs the following steps:
    1. Embeds the query using Jina AI embeddings
    2. Searches for relevant chunks using OpenSearch hybrid search (BM25 + KNN)
    3. Generates an answer using Groq LLM with the retrieved context
    4. Returns the answer with source citations
    
    Args:
        request: RAG query parameters including query text, filters, and LLM settings
        opensearch_client: OpenSearch client (injected)
        embeddings_client: Jina embeddings client (injected)
        groq_client: Groq LLM client (injected)
        
    Returns:
        RAGQueryResponse: Answer with sources, metadata, and token usage
        
    Raises:
        HTTPException: If embedding, search, or generation fails
    """
    try:
        logger.info(f"RAG query received: '{request.query[:100]}...'")
        
        # Step 1: Generate query embedding
        logger.info("Generating query embedding...")
        try:
            query_embedding = await embeddings_client.embed_query(query=request.query)
            logger.info(f"Query embedded successfully (dim: {len(query_embedding)})")
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate query embedding: {str(e)}"
            )
        
        # Step 2: Search for relevant chunks using hybrid search
        logger.info(f"Searching for top {request.top_k} chunks...")
        try:
            search_results = opensearch_client.search_chunks_hybrid(
                query_text=request.query,
                query_embedding=query_embedding,
                min_score=request.min_score,
                categories=request.categories,
                top_k=request.top_k
            )
            
            num_hits = len(search_results['hits'])
            logger.info(f"Found {search_results['total']} total hits, retrieved {num_hits} chunks")
            
            if num_hits == 0:
                logger.warning("No relevant chunks found for query")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="No relevant documents found for your query. Try adjusting your search terms or lowering the min_score threshold."
                )
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to search chunks: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to search documents: {str(e)}"
            )
        
        # Step 3: Generate RAG answer using LLM
        logger.info("Generating answer with Groq LLM...")
        try:
            rag_result = await groq_client.generate_rag_answer(
                query=request.query,
                search_results=search_results['hits'],
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            logger.info(f"Answer generated successfully (model: {rag_result['model']})")
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate answer: {str(e)}"
            )
        
        # Step 4: Format response
        sources = [
            SearchResultSource(**source)
            for source in rag_result['sources']
        ]
        
        response = RAGQueryResponse(
            query=rag_result['query'],
            answer=rag_result['answer'],
            sources=sources,
            num_sources=rag_result['num_sources'],
            model=rag_result['model'],
            finish_reason=rag_result['finish_reason']
        )
        
        logger.info(f"RAG query completed successfully with {len(sources)} sources")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in RAG query: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )


@router.post("/ask", response_model=AskResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a question (simple Q&A, no RAG)",
    description=(
        "Ask a question directly to the LLM without retrieving documents from the database. "
        "This is a simple question-answering endpoint that doesn't use RAG or search. "
        "Useful for general questions that don't require research paper context."
    )
)
async def ask_question(request: AskRequest,
groq_client: Annotated[GroqClient, Depends(get_groq_client)]
) -> AskResponse:
    """Ask a question directly to the LLM without RAG.
    
    Args:
        request: Ask request with question and optional system prompt
        groq_client: Groq LLM client (injected) 
    Returns:
        AskResponse: Answer from the LLM
    Raises:
        HTTPException: If generation fails
    """
    try:
        logger.info(f"Ask question received: '{request.question[:100]}...'")
        
        # Generate answer directly using LLM
        try:
            result = await groq_client.generate_answer(
                query=request.question,
            )
            logger.info(f"Answer generated successfully (model: {result['model']})")
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to generate answer: {str(e)}"
            )
        
        # Format response
        response = AskResponse(
            question=request.question,
            answer=result['answer'],
            model=result['model'],
            finish_reason=result['finish_reason']
        )
        
        logger.info("Ask question completed successfully")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in ask endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Health check for RAG services",
    description="Check if all RAG services (OpenSearch, Jina, Groq) are healthy and accessible."
)
async def health_check(
    opensearch_client: Annotated[OpenSearchClient, Depends(get_opensearch_client)],
    embeddings_client: Annotated[JinaEmbeddingsClient, Depends(get_embeddings_client)],
    groq_client: Annotated[GroqClient, Depends(get_groq_client)]
) -> dict:
    """Check health of all RAG services.
    
    Returns:
        Dictionary with health status of each service
    """
    try:
        # Check OpenSearch
        opensearch_healthy = opensearch_client.client.ping()
        
        # Check Jina (simple embed test)
        try:
            await embeddings_client.embed_query("test")
            jina_healthy = True
        except:
            jina_healthy = False
        
        # Check Groq
        groq_healthy = await groq_client.health_check()
        
        all_healthy = opensearch_healthy and jina_healthy and groq_healthy
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "services": {
                "opensearch": "healthy" if opensearch_healthy else "unhealthy",
                "jina_embeddings": "healthy" if jina_healthy else "unhealthy",
                "groq_llm": "healthy" if groq_healthy else "unhealthy"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )
