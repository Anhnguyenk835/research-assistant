from typing import List, Optional
from pydantic import BaseModel, Field

from schemas.parser.parser_models import PaperProv

class ChunkMetadata(BaseModel):
    """ Metadata for a text chunk. """
    chunk_id: str = Field(..., description="Unique identifier for the chunk")
    start_char: int = Field(..., description="Starting character index in the original text")
    end_char: int = Field(..., description="Ending character index in the original text")
    word_count: int = Field(..., description="Number of words in the chunk")
    section_heading: Optional[str] = Field(None, description="Heading of the section this chunk belongs to")
    prov: Optional[List[PaperProv]] = Field(None, description="Provenance information for the chunk")



class PaperChunk(BaseModel):
    """ Represents a chunk of text with associated metadata. """
    text: str = Field(..., description="The text content of the chunk")
    metadata: ChunkMetadata = Field(..., description="Metadata associated with the chunk")
    arxiv_id: str = Field(..., description="The arXiv ID of the source paper, if applicable")
