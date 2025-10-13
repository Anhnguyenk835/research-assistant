"""Pydantic models for RAG endpoints."""

from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field


class UsageInfo(BaseModel):
    """Token usage information from LLM."""
    
    prompt_tokens: Optional[int] = Field(None, description="Number of tokens in the prompt")
    completion_tokens: Optional[int] = Field(None, description="Number of tokens in the completion")
    total_tokens: Optional[int] = Field(None, description="Total number of tokens used")


class BoundingBox(BaseModel):
    """Bounding box coordinates."""
    
    l: float = Field(..., description="Left x-coordinate")
    t: float = Field(..., description="Top y-coordinate")
    r: float = Field(..., description="Right x-coordinate")
    b: float = Field(..., description="Bottom y-coordinate")
    coord_origin: str = Field(..., description="Coordinate origin (e.g., 'pdf')")


class ProvenanceInfo(BaseModel):
    """Provenance information for a text chunk."""
    
    page_no: int = Field(..., description="Page number where the text is located")
    bbox: BoundingBox = Field(..., description="Bounding box of the text on the page")
    charspan: Optional[List[float]] = Field(None, description="Character span in the text")

class SearchResultSource(BaseModel):
    """Source information from search results."""
    inline_index: int = Field(..., description="Index of the source in the inline list")
    rank: int = Field(..., description="Rank of the source in search results")
    arxiv_id: str = Field(..., description="ArXiv ID of the paper")
    title: str = Field(..., description="Title of the paper")
    section: str = Field(..., description="Section heading where the text was found")
    score: float = Field(..., description="Search relevance score")
    chunk_id: str = Field(..., description="ID of the chunk in the index")
    prov: Optional[List[ProvenanceInfo]] = Field(None, description="Provenance information (page numbers and bounding boxes)")
    chunk_text: Optional[str] = Field(None, description="The actual text content of the chunk")

class RAGQueryRequest(BaseModel):
    """Request model for RAG query endpoint."""
    
    query: str = Field(..., description="The question or query to answer", min_length=1)
    categories: Optional[List[str]] = Field(None, description="Optional list of arXiv categories to filter by")
    top_k: int = Field(5, description="Number of top results to retrieve", ge=1, le=20)
    min_score: float = Field(0.01, description="Minimum relevance score threshold", ge=0.0, le=1.0)
    temperature: float = Field(0.3, description="LLM sampling temperature", ge=0.0, le=2.0)
    max_tokens: int = Field(2048, description="Maximum tokens in LLM response", ge=128, le=4096)
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the main contributions of transformer architecture?",
                "categories": ["cs.AI", "cs.LG"],
                "top_k": 5,
                "min_score": 0.01,
                "temperature": 0.3,
                "max_tokens": 2048
            }
        }

class RAGQueryResponse(BaseModel):
    """Response model for RAG query endpoint."""
    answer: str = Field(..., description="Generated answer with inline source citations (e.g. [1], [2],...)")
    query: str = Field(..., description="The original query")
    num_sources: int = Field(..., description="Number of sources used")
    model: str = Field(..., description="LLM model used to generate the answer")
    usage: dict = Field(..., description="Token usage information from the LLM")
    finish_reason: str = Field(..., description="Reason the generation finished")
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the main contributions of transformer architecture?",
                "answer": "The transformer architecture introduced several key innovations [Source 1]...",
                "sources": [
                    {
                        "inline_index": 0,
                        "rank": 1,
                        "arxiv_id": "1706.03762",
                        "title": "Attention Is All You Need",
                        "section": "Introduction",
                        "score": 0.95,
                        "chunk_id": "doc_1_chunk_5",
                        "prov": [
                            {
                                "page_no": 2,
                                "bbox": {
                                    "l": 100.5,
                                    "t": 200.3,
                                    "r": 450.7,
                                    "b": 250.8,
                                    "coord_origin": "pdf"
                                },
                                "charspan": [0, 150]
                            }
                        ],
                        "chunk_text": "The transformer model..."
                    }
                ],
                "num_sources": 5,
                "model": "llama-3.1-70b-versatile",
                "finish_reason": "stop"
            }
        }


class RAGResponse(BaseModel):
    """Generic response model for RAG endpoints."""
    response: str = Field(..., description="Generated response from the LLM, contain inline citation markers (e.g [1], [2])")
    sources: List[SearchResultSource] = Field(..., description="List of sources used to generate the response")

class AskRequest(BaseModel):
    """Request model for simple Q&A endpoint (no RAG)."""
    
    question: str = Field(..., description="The question to ask the LLM", min_length=1)
    
class AskResponse(BaseModel):
    """Response model for simple Q&A endpoint."""
    
    question: str = Field(..., description="The original question")
    answer: str = Field(..., description="Generated answer from the LLM")
    model: str = Field(..., description="LLM model used to generate the answer")
    finish_reason: str = Field(..., description="Reason the generation finished")
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is machine learning?",
                "answer": "Machine learning is a subset of artificial intelligence that focuses on...",
                "model": "llama-3.1-70b-versatile",
                "finish_reason": "stop"
            }
        }
