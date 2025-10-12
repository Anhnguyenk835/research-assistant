import logging
from typing import List, Optional, Dict
from pathlib import Path

from schemas.parser.parser_models import ParsedPaper
from schemas.indexing.indexing_models import PaperChunk, ChunkMetadata
from schemas.arxiv.arxiv_models import ArxivPaper

from services.embedding_service.jina_client import JinaEmbeddingsClient
from services.opensearch_service.opensearch_client import OpenSearchClient

from .chunker import HeadingChunk


logger = logging.getLogger(__name__)

class HybridIndexingService:
    """ Service to handle hybrid indexing of parsed papers and arXiv metadata. """

    def __init__(self, chunker: HeadingChunk, embedding_client: JinaEmbeddingsClient, opensearch_client: OpenSearchClient):
        """ Initialize the indexing service with a chunker and embedding client. """
        self.chunker = chunker
        self.embedding_client = embedding_client
        self.opensearch_client = opensearch_client

        logger.info("Hybrid Indexing Service initialized")

    async def index_parsed_paper(self, paper_data: ParsedPaper) -> Dict[str, int]:
        """ Index a parsed paper along with optional arXiv metadata."""

        arxiv_id = paper_data.metadata.arxiv_id

        if not arxiv_id:
            logger.warning("Parsed paper does not have an arXiv ID. Skipping indexing.")
            return {"chunks_created": 0, "chunks_indexed": 0, "embeddings_generated": 0, "errors": 1}

        try:
            # Step 1: Chunk the parsed paper
            chunks: List[PaperChunk] = self.chunker.chunk_paper(paper_data)

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
                    "chunk_text": chunk.text, 
                    "chunk_metadata": {
                        "chunk_id": chunk.metadata.chunk_id,
                        "start_char": chunk.metadata.start_char,
                        "end_char": chunk.metadata.end_char,
                        "chunk_word_count": chunk.metadata.word_count,
                        "section_heading": chunk.metadata.section_heading,
                        "prov": [prov.model_dump() for prov in chunk.metadata.prov] if chunk.metadata.prov else [],
                    },
                    "arxiv_metadata": {
                        "arxiv_id": paper_data.metadata.arxiv_id,
                        "title": paper_data.metadata.title,
                        "authors": paper_data.metadata.authors,
                        "abstract": paper_data.metadata.abstract,
                        "categories": paper_data.metadata.categories,
                        "published_date": paper_data.metadata.published_date,
                        "pdf_url": paper_data.metadata.pdf_url
                    }
                }
                chunks_with_embeddings.append({"chunk_data": chunk_data, "embedding": embedding})

            # Step 4: Index chunks with embeddings to OpenSearch
            results = self.opensearch_client.bulk_index_chunks(chunks_with_embeddings)

            logger.info(f"Indexed paper {arxiv_id}: {results['success']} chunks successful, {results['failed']} failed")

            return {
                "chunks_created": len(chunks),
                "chunks_indexed": results["success"],
                "embeddings_generated": len(embeddings),
                "errors": results["failed"],
            }
        
        except Exception as e:
            logger.error(f"Error indexing paper {arxiv_id}: {e}")
            return {"chunks_created": 0, "chunks_indexed": 0, "embeddings_generated": 0, "errors": 1}
        
    async def index_parsed_paper_batch(self, papers: List[ParsedPaper], replace_existing: bool = False) -> Dict[str, int]:
        """Index multiple papers in batch.

        :param papers: List of parsed paper data
        :param replace_existing: If True, delete existing chunks before indexing
        :returns: Aggregated statistics
        """
        total_stats = {
            "papers_processed": 0,
            "total_chunks_created": 0,
            "total_chunks_indexed": 0,
            "total_embeddings_generated": 0,
            "total_errors": 0,
        }

        for paper in papers:
            arxiv_id = paper.metadata.arxiv_id

            if replace_existing and arxiv_id:
                self.opensearch_client.delete_paper_chunks(arxiv_id)
                logger.info(f"Deleted existing chunks for paper {arxiv_id}")

            # index paper
            stats = await self.index_parsed_paper(paper)
            # Update totals
            total_stats["papers_processed"] += 1
            total_stats["total_chunks_created"] += stats["chunks_created"]
            total_stats["total_chunks_indexed"] += stats["chunks_indexed"]
            total_stats["total_embeddings_generated"] += stats["embeddings_generated"]
            total_stats["total_errors"] += stats["errors"]

        logger.info(
            f"Batch indexing complete: {total_stats['papers_processed']} papers, "
            f"{total_stats['total_chunks_indexed']} chunks indexed"
        )

        return total_stats

    async def reindex_paper(self, arxiv_id: str, paper: ParsedPaper) -> Dict[str, int]:
        """Reindex a paper by deleting old chunks and creating new ones.

        :param arxiv_id: ArXiv ID of the paper
        :param paper_data: Updated paper data
        :returns: Indexing statistics
        """
        # Delete existing chunks
        deleted = self.opensearch_client.delete_paper_chunks(arxiv_id)
        if deleted:
            logger.info(f"Deleted existing chunks for paper {arxiv_id}")

        # Index with new data
        logger.info(f"Reindexing paper {arxiv_id}")
        return await self.index_parsed_paper(paper)

        