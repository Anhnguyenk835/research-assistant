import asyncio
from pathlib import Path
import pandas as pd
from schemas.parser.parser_models import ParsedPaper
from schemas.arxiv.arxiv_models import ArxivPaper
from schemas.indexing.indexing_models import PaperChunk

from services.arxiv_service.instance import make_arxiv_client
from services.parser_service.instance import get_pdf_parser_service
from services.indexing_service.instance import make_hybrid_indexing_service
from services.opensearch_service.instance import make_opensearch_client_fresh
from services.embedding_service.instance import make_embeddings_client
from config import get_settings

from typing import List

if __name__ == "__main__":
    setting = get_settings()
    
    # ... existing code for fetching, parsing, and indexing papers ...
    
    # ===== TEST HYBRID SEARCH =====
    print("\n" + "="*80)
    print("TESTING HYBRID SEARCH")
    print("="*80 + "\n")
    
    # Initialize clients
    opensearch_client = make_opensearch_client_fresh(settings=setting)
    embeddings_client = make_embeddings_client(settings=setting)
    
    # 0. Setup indices and RRF pipeline (required for hybrid search)
    print("0. Setting up OpenSearch indices and RRF pipeline...")
    
    # First, check if index exists and verify its mapping
    index_stats = opensearch_client.get_index_stats()
    print(f"   Index exists: {index_stats['exists']}, Documents: {index_stats.get('document_count', 0)}")
    
    if index_stats['exists']:
        # Verify the embedding field is knn_vector
        try:
            mapping = opensearch_client.client.indices.get_mapping(index=opensearch_client.index_name)
            embedding_field = mapping[opensearch_client.index_name]["mappings"]["properties"].get("embedding")
            
            if embedding_field and embedding_field.get("type") == "knn_vector":
                print("   ✓ Index has correct knn_vector mapping")
                force_recreate = False
            else:
                print(f"   ✗ Index has WRONG mapping for embedding field: {embedding_field}")
                print(f"   ⚠️  WARNING: Index has {index_stats.get('document_count', 0)} documents")
                print("   → Recreating index will DELETE ALL DATA!")
                
                # Safety: Ask for confirmation if there's data
                if index_stats.get('document_count', 0) > 0:
                    response = input("   Do you want to proceed? (yes/no): ")
                    force_recreate = response.lower() == 'yes'
                else:
                    force_recreate = True
        except Exception as e:
            print(f"   ✗ Error checking mapping: {e}")
            force_recreate = True
    else:
        force_recreate = False
    
    # Create/recreate indices with correct mapping
    setup_results = opensearch_client.setup_indices(force=force_recreate)
    if setup_results["hybrid_index"]:
        print("   ✓ Hybrid index created/recreated")
    else:
        print("   ✓ Hybrid index already exists with correct mapping")
    
    if setup_results["rrf_pipeline"]:
        print("   ✓ RRF pipeline created")
    else:
        print("   ✓ RRF pipeline already exists")
    print()
    
    # Test query
    test_query = "How data was generated"
    print(f"Query: '{test_query}'\n")
    
    # 1. Get embedding for the query
    print("1. Generating query embedding...")
    query_embedding = asyncio.run(embeddings_client.embed_query(query=test_query))
    print(f"   Embedded query into vector successfully.")
    print(f"   Embedding dimension: {len(query_embedding)}\n")
    
    # 2. Perform hybrid search
    print("2. Executing hybrid search...")
    hybrid_results = opensearch_client.search_chunks_hybrid(
        query_text=test_query,
        query_embedding=query_embedding,
        min_score=0.01,  # Lower threshold to get more results
        categories=None,  # Or specify like ["cs.AI", "cs.LG"]
        top_k=5
    )
    
    print(f"   Total hits: {hybrid_results['total']}")
    print(f"   Results returned: {len(hybrid_results['hits'])}\n")
    
    # 3. Display results
    print("3. Hybrid Search Results:")
    print("-" * 80)
    for i, hit in enumerate(hybrid_results['hits'], 1):
        print(f"\n[Result {i}]")
        print(f"RRF Score: {hit['score']:.4f}")
        print(f"Chunk ID: {hit['chunk_id']}")
        
        # Access nested metadata fields
        chunk_data = hit['chunk']
        arxiv_meta = chunk_data.get('arxiv_metadata', {})
        chunk_meta = chunk_data.get('chunk_metadata', {})
        
        print(f"ArXiv ID: {arxiv_meta.get('arxiv_id', 'N/A')}")
        print(f"Title: {arxiv_meta.get('title', 'N/A')}")
        print(f"Section: {chunk_meta.get('section_heading', 'N/A')}")
        print(f"Prov: {chunk_meta.get('prov', 'N/A')}")
        print(f"Word Count: {chunk_meta.get('word_count', 0)}")
        print(f"Content: {chunk_data.get('chunk_text', 'N/A')}")
        
        # Show highlights if available
        if hit.get('highlight'):
            print(f"Highlights: {hit['highlight']}")
        print("-" * 80)
    
    # 4. Save results to CSV for analysis
    print("\n\n4. Saving results to CSV...")
    
    # Helper function to safely extract data
    def extract_hit_data(hit):
        chunk_data = hit['chunk']
        arxiv_meta = chunk_data.get('arxiv_metadata', {})
        chunk_meta = chunk_data.get('chunk_metadata', {})
        
        return {
            "rank": None,  # Will be set later
            "score": hit['score'],
            "chunk_id": hit['chunk_id'],
            "arxiv_id": arxiv_meta.get('arxiv_id', 'N/A'),
            "title": arxiv_meta.get('title', 'N/A'),
            "authors": ', '.join(arxiv_meta.get('authors', [])),
            "categories": ', '.join(arxiv_meta.get('categories', [])),
            "section": chunk_meta.get('section_heading', 'N/A'),
            "prov": chunk_meta.get('prov', 'N/A'),
            "chunk_index": chunk_meta.get('chunk_id', 'N/A'),
            "word_count": chunk_meta.get('word_count', 0),
            "start_char": chunk_meta.get('start_char', 0),
            "end_char": chunk_meta.get('end_char', 0),
            "content": chunk_data.get('chunk_text', 'N/A'),
            "highlight": str(hit.get('highlight', {}))
        }
    
    # Create DataFrame from hybrid results
    df_results = pd.DataFrame([
        extract_hit_data(hit) 
        for hit in hybrid_results['hits']
    ])
    df_results['rank'] = range(1, len(df_results) + 1)
    df_results.to_csv("hybrid_search_results.csv", index=False)
    print(f"   Results saved to: hybrid_search_results.csv")
    
    # 5. Display statistics
    print("\n\n5. Search Statistics:")
    print("-" * 80)
    if hybrid_results['hits']:
        avg_score = sum(h['score'] for h in hybrid_results['hits']) / len(hybrid_results['hits'])
        max_score = max(h['score'] for h in hybrid_results['hits'])
        min_score = min(h['score'] for h in hybrid_results['hits'])
        
        print(f"Total Hits: {hybrid_results['total']}")
        print(f"Results Returned: {len(hybrid_results['hits'])}")
        print(f"Average Score: {avg_score:.4f}")
        print(f"Max Score: {max_score:.4f}")
        print(f"Min Score: {min_score:.4f}")
    else:
        print("No results found.")
    
    print("\n" + "="*80)
    print("HYBRID SEARCH TEST COMPLETED")
    print("="*80)