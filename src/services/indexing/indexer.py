import logging
from typing import List, Optional, Dict
from pathlib import Path

from mistralai_gcp import TextChunk

from .chunker import HeadingChunk
from schemas.parser.parser_models import ParsedPaper
from schemas.indexing.indexing_models import PaperChunk, ChunkMetadata
from src.schemas.arxiv.arxiv_models import ArxivPaper

logger = logging.getLogger(__name__)

class HybridIndexingService:
    """ Service to handle hybrid indexing of parsed papers and arXiv metadata. """

    def __init__(self, chunker: HeadingChunk):
        self.chunker = chunker

        logger.info("Hybrid Indexing Service initialized")

    async def index_parsed_paper(self, parsed_paper: ParsedPaper, arxiv_metadata: Optional[ArxivPaper] = None) -> List[TextChunk]:
        """ Index a parsed paper along with optional arXiv metadata."""

        arxiv_id = parsed_paper.metadata.arxiv_id

        if not arxiv_id:
            logger.warning("Parsed paper does not have an arXiv ID. Skipping indexing.")
            return []

        try:
            chunks = self.chunker.chunk_paper(parsed_paper)

        except Exception as e:
            logger.error(f"Error during chunking: {e}")
            return []

        