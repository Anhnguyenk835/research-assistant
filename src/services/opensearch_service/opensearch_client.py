import logging
from typing import List, Dict, Any, Optional

from opensearchpy import OpenSearch, helpers
from config import Settings
from .query_builder import QueryBuilder
from .index_config import ARXIV_PAPERS_CHUNKS_MAPPING, HYBRID_RRF_PIPELINE

logger = logging.getLogger(__name__)

class OpenSearchClient:
    """OpenSearch client supporting BM25 and hybrid search with native RRF."""

    def __init__(self, host: str, settings: Settings):
        """Initialize OpenSearch client.

        :param host: OpenSearch host URL
        :param settings: Application settings
        """

        self.host = host
        self.settings = settings
        self.index_name = f"{settings.opensearch.index_name}-{settings.opensearch.chunk_index_suffix}"

        # Build client configuration
        client_config = {
            "hosts": [host],
            "use_ssl": True,
            "verify_certs": False,
            "ssl_show_warn": False,
        }

        # Add authentication if credentials are provided
        if settings.opensearch.username and settings.opensearch.password:
            client_config["http_auth"] = (
                settings.opensearch.username,
                settings.opensearch.password
            )
            logger.info(f"OpenSearch client initialized with authentication for user: {settings.opensearch.username}")
        else:
            logger.warning("OpenSearch client initialized without authentication credentials")

        self.client = OpenSearch(**client_config)

        logger.info(f"OpenSearch client initialized with host: {host}")

    def health_check(self) -> bool:
        """Check OpenSearch cluster health."""
        try:
            health = self.client.cluster.health()
            status = health.get("status", "red")
            logger.info(f"OpenSearch cluster health status: {status}")
            return status in ["green", "yellow"]
        except Exception as e:
            logger.error(f"OpenSearch health check failed: {e}")
            return False
        
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics for the configured index."""
        try:
            if not self.client.indices.exists(index=self.index_name):
                logger.info(f"Index {self.index_name} does not exist.")  # Changed from WARNING to INFO
                return {"index_name": self.index_name, "exists": False, "document_count": 0}
            
            stats_response = self.client.indices.stats(index=self.index_name)
            index_stats = stats_response["indices"][self.index_name]["total"]

            return {
                "index_name": self.index_name,
                "exists": True,
                "document_count": index_stats["docs"]["count"],
                "deleted_count": index_stats["docs"]["deleted"],
                "size_in_bytes": index_stats["store"]["size_in_bytes"],
            }
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")
            return {"index_name": self.index_name, "exists": False, "document_count": 0, "error": str(e)}
        
    def setup_indices(self, force: bool = False) -> Dict[str, bool]:
        """Setup the hybrid search index and RRF pipeline."""
        results = {}
        results["hybrid_index"] = self._create_hybrid_index(force=force)
        results["rrf_pipeline"] = self._create_rrf_pipeline(force=force)
        return results
    
    def _create_hybrid_index(self, force: bool = False) -> bool:
        """Create hybrid index for all search types (BM25, vector search, hybrid search)
        
        :param force: If True, recreate index even if it exists
        :returns: True if created, False if already exists
        """
        try:
            if force and self.client.indices.exists(index=self.index_name):
                self.client.indices.delete(index=self.index_name)
                logger.info(f"Deleted existing index {self.index_name} due to force=True")
            
            if not self.client.indices.exists(index=self.index_name):
                self.client.indices.create(
                    index=self.index_name,
                    body=ARXIV_PAPERS_CHUNKS_MAPPING
                )
                logger.info(f"Created index {self.index_name}")
                return True
            
            logger.info(f"Index {self.index_name} already exists")
            return False
        
        except Exception as e:
            logger.error(f"Failed to create hybrid index: {e}")
            raise

    def _create_rrf_pipeline(self, force: bool = False) -> bool:
        """Create RRF pipeline for hybrid search.

        :param force: If True, recreate pipeline even if it exists
        :returns: True if created, False if already exists
        """
        try: 
            pipeline_id = HYBRID_RRF_PIPELINE["id"]

            if force:
                try:
                    self.client.ingest.get_pipeline(id=pipeline_id)
                    self.client.ingest.delete_pipeline(id=pipeline_id)
                    logger.info(f"Deleted existing RRF pipeline {pipeline_id} due to force=True")
                except Exception as e:
                    logger.warning(f"Failed to delete existing RRF pipeline {pipeline_id}: {e}")
                    pass
            
            try:
                self.client.ingest.get_pipeline(id=pipeline_id)
                logger.info(f"RRF pipeline {pipeline_id} already exists")
                return False
            except Exception:
                pass  # Pipeline does not exist, proceed to create

            pipeline_body = {
                "description": HYBRID_RRF_PIPELINE["description"],
                "phase_results_processors": HYBRID_RRF_PIPELINE["phase_results_processors"]
            }

            self.client.transport.perform_request(
                method="PUT",
                url=f"/_search/pipeline/{pipeline_id}",
                body=pipeline_body
            )

            logger.info(f"Created RRF pipeline {pipeline_id}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to create RRF pipeline: {e}")
            raise

    def search_chunks_vector(self, query_embedding: List[float], categories: Optional[List[str]] = None, top_k: int = 10) -> Dict[str, Any]:
        """Perform vector search on chunks using cosine similarity.

        :param query_embedding: Embedding vector for the search query
        :param top_k: Number of top results to return
        :param categories: Optional categories to filter search results
        :returns: Search results from OpenSearch
        """
        try:
            # filter fisrt
            filter_clause = []
            if categories:
                filter_clause.append({
                    "terms": {
                        "arxiv_metadata.categories": categories
                    }
                })
            
            query_body = {
                "size": top_k,
                "query": {
                    "knn": {
                        "embedding": {
                            "vector": query_embedding,
                            "k": top_k
                        }
                    }
                }
            }

            if filter_clause:
                query_body["query"] = {
                    "bool": {
                        "must": query_body["query"],
                        "filter": filter_clause
                    }
                }
            
            response = self.client.search(index=self.index_name, body=query_body)

            results = {
                "total": response["hits"]["total"]["value"],
                "hits": [
                    {
                        "chunk": hit["_source"],
                        "score": hit["_score"],
                        "chunk_id": hit["_id"]
                    }
                    for hit in response["hits"]["hits"]
                ]
            }

            logger.info(f"Vector search returned {len(results['hits'])} hits out of {results['total']} total")
            return results
        
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return {"total": 0, "hits": []}

    def search_chunks_bm25(self, query_text: str, _from: int, latest: bool, categories: Optional[List[str]] = None, top_k: int = 10) -> Dict[str, Any]:
        """Perform BM25 text search on chunks.

        :param query_text: Text query for BM25 search
        :param _from: Starting index for search results
        :param latest: Whether to include only the latest results
        :param categories: Optional categories to filter search results
        :param top_k: Number of top results to return
        :returns: Search results from OpenSearch
        """
        builder = QueryBuilder(
            query=query_text,
            size=top_k,
            _from=_from,
            latest_papers=latest,
            categories=categories,
            search_chunks=True,  # Enable chunk search mode
        )
        query_body = builder.build()

        response = self.client.search(index=self.index_name, body=query_body)

        results = {
            "total": response["hits"]["total"]["value"],
            "hits": [
                {
                    "chunk": hit["_source"],
                    "score": hit["_score"],
                    "chunk_id": hit["_id"],
                    "highlight": hit["highlight"] if "highlight" in hit else {}
                }
                for hit in response["hits"]["hits"]
            ]
        }

        logger.info(f"BM25 search returned {len(results['hits'])} hits out of {results['total']} total")
        return results

    def search_hybrid_native(self, query_text: str, query_embedding: List[float], min_score: float, categories: Optional[List[str]] = None, top_k: int = 10) -> Dict[str, Any]:
        """Perform hybrid search using native RRF pipeline.

        :param query_embedding: Embedding vector for the search query
        :param query_text: Text query for BM25 search
        :param min_score: Minimum score threshold for results
        :param categories: Optional categories to filter search results
        :param top_k: Number of top results to return
        :returns: Search results from OpenSearch
        """
        builder = QueryBuilder(
            query=query_text,
            size=top_k * 2,  # Fetch more to allow for filtering by min_score
            _from=0,
            latest_papers=False,
            categories=categories,
            search_chunks=True,  # Enable chunk search mode
        )

        bm25_search_body = builder.build()
        bm25_query = bm25_search_body["query"]

        hybrid_query = {
            "hybrid": {
                "queries": [
                    bm25_query, 
                    {"knn": 
                        {"embedding": 
                            {"vector": query_embedding, 
                            "k": top_k * 2
                            }
                        }
                    }
                ]
            }
        }

        query_body = {
            "size": top_k,
            "query": hybrid_query,
            "_source": bm25_search_body["_source"],
            "highlight": bm25_search_body["highlight"],
        }

        if categories:
            query_body["query"] = {
                "bool": {
                    "must": hybrid_query,
                    "filter": [{"terms": {"arxiv_metadata.categories": categories}}]
                }
            }

        # Execute search with RRF pipeline
        response = self.client.search(
            index=self.index_name, 
            body=query_body, 
            params={
                "search_pipeline": HYBRID_RRF_PIPELINE["id"]
            }
        )

        results = {
            "total": response["hits"]["total"]["value"],
            "hits": [
                {
                    "chunk": hit["_source"],
                    "score": hit["_score"],
                    "chunk_id": hit["_id"],
                    "highlight": hit["highlight"] if "highlight" in hit else {}
                }
                for hit in response["hits"]["hits"]
                if hit["_score"] >= min_score
            ]
        }

        logger.info(f"Hybrid search returned {len(results['hits'])} hits out of {results['total']} total")
        return results
    
    def search_chunks_hybrid(self, query_text: str, query_embedding: List[float], min_score: float, categories: Optional[List[str]] = None, top_k: int = 10) -> Dict[str, Any]:
        """Hybrid search combining BM25 and vector similarity using native RRF."""
        return self.search_hybrid_native(
            query_text=query_text,
            query_embedding=query_embedding,
            min_score=min_score,
            categories=categories,
            top_k=top_k
        )
    
    def index_chunks(self, chunk_data: Dict[str, Any], embedding: List[float]) -> bool:
        """Index a chunk of data with its corresponding embedding."""
        try:
            chunk_data["embedding"] = embedding
            response = self.client.index(
                index=self.index_name,
                body=chunk_data,
                refresh=True  # Ensure the document is searchable immediately
            )
            return response["result"] in ["created", "updated"]
        except Exception as e:
            logger.error(f"Error indexing chunk: {e}")
            return False
        
    def bulk_index_chunks(self, chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        """Bulk index multiple chunks of data.

        :param chunks: List of chunk data dictionaries, each containing 'chunk_data' and 'embedding'
        :returns: Summary of indexing results
        """
        try: 
            actions = []
            for item in chunks:
                chunk_data = item["chunk_data"].copy()
                chunk_data["embedding"] = item["embedding"]
                action = {
                    "_index": self.index_name,
                    "_source": chunk_data
                }
                actions.append(action)

            success, failed = helpers.bulk(self.client, actions, refresh=True)

            logger.info(f"Bulk indexed {success} chunks with {len(failed)} failures")
            return {"success": success, "failed": len(failed)}
        
        except Exception as e:
            logger.error(f"Bulk indexing failed: {e}")
            raise

    def delete_paper_chunks(self, arxiv_id: str) -> bool:
        """Delete all chunks for a specific paper.

        :param arxiv_id: ArXiv ID of the paper
        :returns: True if deletion was successful
        """
        try:
            response = self.client.delete_by_query(
                index=self.index_name, 
                body={
                    "query": {
                            "term": {
                                "arxiv_metadata.arxiv_id": arxiv_id
                            }
                        }
                    }, 
                refresh=True
            )

            deleted = response.get("deleted", 0)
            logger.info(f"Deleted {deleted} chunks for paper {arxiv_id}")
            return deleted > 0

        except Exception as e:
            logger.error(f"Error deleting chunks: {e}")
            return False
        
    def get_chunks_by_paper(self, arxiv_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a specific paper.

        :param arxiv_id: ArXiv ID of the paper
        :returns: List of chunks sorted by chunk_index
        """
        try:
            search_body = {
                "query": {
                    "term": {
                        "arxiv_metadata.arxiv_id": arxiv_id
                    }
                },
                "size": 1000,
                "sort": [{"chunk_metadata.chunk_id": "asc"}],
                "_source": {"excludes": ["embedding"]},
            }

            response = self.client.search(index=self.index_name, body=search_body)

            chunks = []
            for hit in response["hits"]["hits"]:
                src = hit["_source"]
                src["chunk_id"] = src["chunk_metadata"]["chunk_id"]
                chunks.append(src)

            return chunks

        except Exception as e:
            logger.error(f"Error getting chunks: {e}")
            return []