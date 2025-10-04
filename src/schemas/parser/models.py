from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

class ParserType(str, Enum):
    """ Parser type"""
    DOCLING = "docling"

class PaperBbox(BaseModel):
    """ Represents a bounding box in a paper. """
    l: float = Field(..., description="Left x-coordinate")
    t: float = Field(..., description="Bottom y-coordinate")
    r: float = Field(..., description="Right x-coordinate")
    b: float = Field(..., description="Top y-coordinate")
    coord_origin: str = Field(..., description="Coordinate origin, e.g., 'pdf' or 'image'")

class PaperProv(BaseModel):
    """ Represents provenance information for a paper element. """
    page_no: int = Field(..., description="Page number where the element is located")
    bbox: PaperBbox = Field(..., description="Bounding box of the element")
    charspan: Optional[List[float]] = Field(None, description="Character span in the text")

class PaperSection(BaseModel):
    """ Represents a section of a paper. """
    label: str = Field(..., description="Title of the section")
    content: str = Field(..., description="Content of the section")
    prov: Optional[List[PaperProv]] = Field(None, description="Provenance information for the section")
    level: Optional[int] = Field(None, description="Level of the section in the hierarchy (e.g., 1 for top-level sections)")

class PaperFigure(BaseModel):
    """ Represents a figure in a paper. """
    id: str = Field(..., description="Unique identifier for the figure")
    caption: str = Field(..., description="Caption of the figure")
    # image_data: Optional[bytes] = Field(None, description="Binary data of the figure image")
    page_number: int = Field(..., description="Page number where the figure is located")

class PaperTable(BaseModel):
    """ Represents a table in a paper. """
    label: str = Field(..., description="Title or label of the table")
    prov: List[PaperProv] = Field(..., description="Provenance information for the table")
    content: str = Field(..., description="Content of the table in a structured format (e.g., text or Markdown)")

class PdfContent(BaseModel):
    """ Represents the content extracted from a PDF. """
    sections: List[PaperSection] = Field(default_factory=list, description="List of sections in the paper")
    tables: List[PaperTable] = Field(default_factory=list, description="List of tables in the paper")
    figures: List[PaperFigure] = Field(default_factory=list, description="List of figures in the paper")
    raw_text: str = Field(..., description="Raw text extracted from the PDF")
    references: List[str] = Field(default_factory=list, description="List of references cited in the paper")
    parser_type: ParserType = Field(..., description="Type of parser used to extract the content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the PDF or extraction process")

class ArXivMetadata(BaseModel):
    """ Metadata for an arXiv paper. """
    arxiv_id: str = Field(..., description="arXiv identifier")
    title: str = Field(..., description="Title of the paper")
    authors: List[str] = Field(default_factory=list, description="List of authors")
    abstract: str = Field(..., description="Abstract of the paper")
    categories: List[str] = Field(default_factory=list, description="List of categories")
    published_date: str = Field(..., description="Publication date")
    url: str = Field(..., description="PDF download URL")
    # doi: Optional[str] = Field(None, description="Digital Object Identifier")
    # journal_ref: Optional[str] = Field(None, description="Journal reference if available")
    # comments: Optional[str] = Field(None, description="Comments about the paper")

class ParsedPaper(BaseModel):
    """ Represents a parsed paper with its content and metadata. """
    metadata: ArXivMetadata = Field(..., description="Metadata of the arXiv paper")
    content: Optional[PdfContent] = Field(None, description="Extracted content from the PDF")