from datetime import datetime
from typing import Optional, List, Dict, Any
from uuid import uuid4

from pydantic import BaseModel, Field, HttpUrl, validator

class ArxivPaper(BaseModel):
    """Model representing a paper from arXiv."""
    arxiv_id: str = Field(..., description="The unique identifier for the arXiv paper.")
    title: str = Field(..., description="The title of the paper.")
    authors: List[str] = Field(..., description="List of authors of the paper.")
    abstract: str = Field(..., description="The abstract of the paper.")
    categories: List[str] = Field(..., description="List of categories the paper belongs to.")
    published_date: str = Field(..., description="The publication date of the paper.")
    pdf_url: str = Field(..., description="URL to download the PDF of the paper.")