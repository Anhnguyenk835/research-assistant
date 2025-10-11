import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class QueryBuilder:
    """
    Unified query builder for OpenSearch supporting both paper-level and chunk-level search.

    Builds complex OpenSearch queries with proper scoring, filtering, and highlighting.
    """

    def __init__(
        self,
        query: str,
        size: int = 10,
        from_: int = 0,
        fields: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        track_total_hits: bool = True,
        latest_papers: bool = False,
        search_chunks: bool = False,
    ):
        """Initialize query builder.

        :param query: Search query text
        :param size: Number of results to return
        :param from_: Offset for pagination
        :param fields: Fields to search in (if None, auto-determined based on search_chunks)
        :param categories: Filter by categories
        :param track_total_hits: Whether to track total hits accurately
        :param latest_papers: Sort by publication date instead of relevance
        :param search_chunks: Whether searching chunks (True) or papers (False)
        """
        self.query = query.strip()
        self.size = size
        self.from_ = from_
        self.categories = categories
        self.track_total_hits = track_total_hits
        self.latest_papers = latest_papers
        self.search_chunks = search_chunks

        # weight of fields for scoring
        if fields is None:
            if search_chunks:
                self.fields = [
                    "chunk_text^3", 
                    "arxiv_metadata.title^2", 
                    "arxiv_metadata.abstract^1"
                ]
            else:
                self.fields = [
                    "arxiv_metadata.title^3", 
                    "arxiv_metadata.abstract^2", 
                    "arxiv_metadata.authors^1"
                ]
        else:
            self.fields = fields

    def build(self) -> Dict[str, Any]:
        """Build the complete OpenSearch query.

        :returns: Complete query dictionary ready for OpenSearch
        """
        query_body = {
            "query": self._build_query(),
            "size": self.size,
            "from": self.from_,
            "track_total_hits": self.track_total_hits,
            "_source": self._build_source_fields(),
            "highlight": self._build_highlight(),
        }

        sort = self._build_sort()
        if sort:
            query_body["sort"] = sort

        return query_body

    def _build_query(self) -> Dict[str, Any]:
        """Build the main query with filters.

        :returns: Query dictionary with bool structure
        """
        must_clauses = []

        if self.query.strip():
            must_clauses.append(self._build_text_query())

        filter_clauses = self._build_filters()

        bool_query = {}

        if must_clauses:
            bool_query["must"] = must_clauses
        else:
            bool_query["must"] = [{"match_all": {}}]

        if filter_clauses:
            bool_query["filter"] = filter_clauses

        return {"bool": bool_query}

    def _build_text_query(self) -> Dict[str, Any]:
        """Build the main text search query.

        :returns: Multi-match query for text search
        """
        return {
            "multi_match": {
                "query": self.query,
                "fields": self.fields,  # field weighting
                "type": "best_fields",
                "operator": "or",
                "fuzziness": "AUTO",    # for typo tolerance
                "prefix_length": 2,     # number of initial characters not to fuzzify
                "minimum_should_match": "2<75%",  # at least 2 words or 75% of words should match
            }
        }

    def _build_filters(self) -> List[Dict[str, Any]]:
        """Build filter clauses for the query.

        :returns: List of filter clauses
        """
        filters = []

        if self.categories:
            filters.append(
                {"terms": {
                    "arxiv_metadata.categories": self.categories
                    }
                }
            )

        return filters

    def _build_source_fields(self) -> Any:
        """Define which fields to return in results.

        :returns: Source field configuration (list for papers, dict for chunks)
        """
        if self.search_chunks:
            return {"excludes": ["embedding"]}
        else:
            return {
                "includes": [
                    "arxiv_metadata.arxiv_id", 
                    "arxiv_metadata.title", 
                    "arxiv_metadata.authors", 
                    "arxiv_metadata.abstract", 
                    "arxiv_metadata.categories", 
                    "arxiv_metadata.published_date", 
                    "arxiv_metadata.pdf_url"
                ]
            }

    def _build_highlight(self) -> Dict[str, Any]:
        """Build highlighting configuration.

        :returns: Highlight configuration dictionary
        """
        if self.search_chunks:
            return {
                "fields": {
                    "chunk_text": {
                        "fragment_size": 150,
                        "number_of_fragments": 2,
                        "pre_tags": ["<mark>"],
                        "post_tags": ["</mark>"],
                    },
                    "arxiv_metadata.title": {
                        "fragment_size": 0, 
                        "number_of_fragments": 0, 
                        "pre_tags": ["<mark>"], 
                        "post_tags": ["</mark>"]
                    },
                    "arxiv_metadata.abstract": {
                        "fragment_size": 150,
                        "number_of_fragments": 1,
                        "pre_tags": ["<mark>"],
                        "post_tags": ["</mark>"],
                    },
                },
                "require_field_match": False,
            }
        else:
            # Paper-specific highlighting
            return {
                "fields": {
                    "arxiv_metadata.title": {
                        "fragment_size": 0,
                        "number_of_fragments": 0,
                    },
                    "arxiv_metadata.abstract": {
                        "fragment_size": 150,
                        "number_of_fragments": 3,
                        "pre_tags": ["<mark>"],
                        "post_tags": ["</mark>"],
                    },
                    "arxiv_metadata.authors": {
                        "fragment_size": 0,
                        "number_of_fragments": 0,
                        "pre_tags": ["<mark>"],
                        "post_tags": ["</mark>"],
                    },
                },
                "require_field_match": False,
            }

    def _build_sort(self) -> Optional[List[Dict[str, Any]]]:
        """Build sorting configuration.

        :returns: Sort configuration or None for relevance scoring
        """
        if self.latest_papers:
            return [
                {"arxiv_metadata.published_date": {"order": "desc"}},
                {"_score": {"order": "desc"}}
            ]

        if self.query.strip():
            return None

        return [
            {
                "arxiv_metadata.published_date": {"order": "desc"}
            }, 
            {"_score": {"order": "desc"}}
        ]