"""OpenSearch index configuration for hybrid search (BM25 + Vector).

This configuration supports both keyword search (BM25) and vector similarity search
using HNSW algorithm for approximate nearest neighbor search.
"""

ARXIV_PAPERS_CHUNKS_INDEX = "arxiv-papers-chunks"

# Index mapping for chunked papers with vector embeddings
ARXIV_PAPERS_CHUNKS_MAPPING = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0,
        "index.knn": True,  # Enable KNN at index level
        "analysis": {
            "analyzer": {
                "standard_analyzer": {
                    "type": "standard",
                    "stopwords": "_english_"
                },
                "text_analyzer": {
                    "type": "custom",
                    "tokenizer": "standard",
                    "filter": ["lowercase", "stop", "snowball"]
                }
            }
        },
    },
    "mappings": {
        "dynamic": "strict",
        "properties": {
            # === Chunk-level metadata ===
            "chunk_metadata": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "keyword"},
                    "start_char": {"type": "integer"},
                    "end_char": {"type": "integer"},
                    "chunk_word_count": {"type": "integer"},
                    "section_heading": {"type": "keyword"},
                    "prov": {
                        "type": "nested",
                        "properties": {
                            "page_no": {"type": "integer"},
                            "bbox": {
                                "type": "object",
                                "enabled": False,  # Disable indexing/searching on bbox
                                "properties": {
                                    "left": {"type": "float"},
                                    "top": {"type": "float"},
                                    "right": {"type": "float"},
                                    "bottom": {"type": "float"}
                                }
                            },
                            "charspan": {"type": "float"}
                        }
                    },
                    "created_at": {"type": "date"},
                    "updated_at": {"type": "date"}
                }
            },

            # === Paper-level metadata ===
            "arxiv_metadata": {
                "type": "object",
                "properties": {
                    "arxiv_id": {"type": "keyword"},
                    "title": {
                        "type": "text",
                        "analyzer": "text_analyzer",
                        "fields": {
                            "keyword": {"type": "keyword", "ignore_above": 256}
                        }
                    },
                    "authors": {
                        "type": "text",
                        "analyzer": "standard_analyzer",
                        "fields": {
                            "keyword": {"type": "keyword", "ignore_above": 256}
                        }
                    },
                    "abstract": {"type": "text", "analyzer": "text_analyzer"},
                    "categories": {"type": "keyword"},
                    "published_date": {"type": "date"},
                    "pdf_url": {"type": "keyword"}
                }
            },

            # === Chunk text content ===
            "chunk_text": {
                "type": "text",
                "analyzer": "text_analyzer",
                "fields": {
                    "keyword": {"type": "keyword", "ignore_above": 256}
                }
            },

            # === Embedding vector for semantic search ===
            "embedding": {
                "type": "knn_vector",
                "dimension": 1024,  # Jina embedding v3
                "method": {
                    "name": "hnsw",
                    "space_type": "cosinesimil",
                    "engine": "lucene",  # Use lucene engine (nmslib is deprecated in OpenSearch 3.0+)
                    "parameters": {
                        "ef_construction": 512,
                        "m": 16
                    }
                }
            }
        }
    }
}


HYBRID_RRF_PIPELINE = {
    "id": "hybrid-rrf-pipeline",
    "description": "Post processor for hybrid RRF search",
    "phase_results_processors": [
        {
            "score-ranker-processor": {
                "combination": {
                    "technique": "rrf",  # Reciprocal Rank Fusion
                    "rank_constant": 60,  # Default k=60 for RRF formula: 1/(k+rank)
                }
            }
        }
    ],
}

# Alternative: Weighted average pipeline (commented out - not used by default)
# This could be used if you need explicit control over BM25 vs vector weights
# However, RRF generally provides better results without manual weight tuning
"""
HYBRID_SEARCH_PIPELINE = {
    "id": "hybrid-ranking-pipeline",
    "description": "Hybrid search pipeline using weighted average for BM25 and vector similarity",
    "phase_results_processors": [
        {
            "normalization-processor": {
                "normalization": {
                    "technique": "l2"  # L2 normalization for better score distribution
                },
                "combination": {
                    "technique": "harmonic_mean",  # Harmonic mean often works better than arithmetic
                    "parameters": {
                        "weights": [0.3, 0.7]  # 30% BM25, 70% vector similarity
                    }
                }
            }
        }
    ]
}
"""