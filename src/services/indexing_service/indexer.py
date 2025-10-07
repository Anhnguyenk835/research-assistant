import logging
from typing import List, Optional, Dict
from pathlib import Path

from .chunker import HeadingChunk
from services.embedding_service.jina_client import JinaEmbeddingsClient
from schemas.parser.parser_models import ParsedPaper
from schemas.indexing.indexing_models import PaperChunk, ChunkMetadata
from schemas.arxiv.arxiv_models import ArxivPaper


logger = logging.getLogger(__name__)

class HybridIndexingService:
    """ Service to handle hybrid indexing of parsed papers and arXiv metadata. """

    def __init__(self, chunker: HeadingChunk, embedding_client: JinaEmbeddingsClient):
        """ Initialize the indexing service with a chunker and embedding client. """
        self.chunker = chunker
        self.embedding_client = embedding_client

        logger.info("Hybrid Indexing Service initialized")

    async def index_parsed_paper(self, paper_data: ParsedPaper) -> Dict[str, int]:
        """ Index a parsed paper along with optional arXiv metadata."""

        arxiv_id = paper_data.metadata.arxiv_id

        if not arxiv_id:
            logger.warning("Parsed paper does not have an arXiv ID. Skipping indexing.")
            return {"chunks_created": 0, "chunks_indexed": 0, "embeddings_generated": 0, "errors": 1}

        try:
            # Step 1: Chunk the parsed paper
            chunks = self.chunker.chunk_paper(paper_data)

            if not chunks:
                logger.warning(f"No chunks created for paper {arxiv_id}. Skipping indexing.")
                return {"chunks_created": 0, "chunks_indexed": 0, "embeddings_generated": 0, "errors": 1}
        
            logger.info(f"Created {len(chunks)} chunks for paper {arxiv_id}")

            # Step 2: Embed the chunks
            chunk_texts = [chunk.text for chunk in chunks]
            embeddings = await self.embedding_client.embed_passages(
                texts=chunk_texts, 
                batch_size=50
            )
            if len(embeddings) != len(chunks):
                logger.error(f"Number of embeddings {len(embeddings)} does not match number of chunks {len(chunks)} for paper {arxiv_id}.")
                return {"chunks_created": len(chunks), "chunks_indexed": 0, "embeddings_generated": len(embeddings), "errors": 1}
            
            # Step 3: Prepare data with chunks and embeddings for indexing with opensearch
            chunks_with_embeddings = []
            for chunk, embedding in zip(chunks, embeddings):
                # prepare data for opensearch
                chunk_data = {
                    "arxiv_id": arxiv_id,
                    "chunk_id": chunk.metadata.chunk_id,
                    "text": chunk.text,
                    "start_char": chunk.metadata.start_char,
                    "end_char": chunk.metadata.end_char,
                    "word_count": chunk.metadata.word_count,
                    "section_heading": chunk.metadata.section_heading,
                    "prov": chunk.metadata.prov,
                    "embedding_model": "jina-embeddings-v3"
                }
                chunks_with_embeddings.append({"chunk_data": chunk_data, "embedding": embedding})

            # Step 4: Index chunks with embeddings to OpenSearch

        except Exception as e:
            logger.error(f"Error during chunking: {e}")
            return {"chunks_created": 0, "chunks_indexed": 0, "embeddings_generated": 0, "errors": 1}

        