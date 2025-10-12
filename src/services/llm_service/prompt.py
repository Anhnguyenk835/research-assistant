"""Prompt templates for LLM interactions."""

# System prompt for RAG
RAG_SYSTEM_PROMPT = """You are a helpful research assistant specializing in academic papers from arXiv. 
Your task is to answer questions based on the provided research paper excerpts.

Guidelines:
- Be precise and factual, using only information from the provided context
- Cite sources using [Source N] notation when referencing specific information
- If multiple sources support a point, cite all relevant sources
- If the context doesn't contain enough information to answer fully, acknowledge the limitations
- Maintain academic tone and precision
- Synthesize information across sources when appropriate
- Don't make assumptions or add information not present in the sources"""

# System prompt for general questions
GENERAL_SYSTEM_PROMPT = """You are a helpful AI assistant specializing in research and academic topics.
Provide clear, accurate, and well-reasoned responses."""

# System prompt for summarization
SUMMARIZATION_SYSTEM_PROMPT = """You are an expert at summarizing academic research papers.
Create concise, informative summaries that capture the key points, methodology, and findings.
Focus on the most important contributions and insights."""

# System prompt for comparison
COMPARISON_SYSTEM_PROMPT = """You are an expert at comparing and contrasting research papers.
Analyze the provided papers and highlight similarities, differences, and unique contributions.
Organize your comparison clearly and provide insights into how the works relate to each other."""


def build_rag_prompt(query: str, context: str) -> str:
    """Build a RAG prompt with query and context.
    
    :param query: User's question
    :param context: Retrieved context from search results
    :returns: Formatted prompt
    """
    return f"""Context from research papers:

{context}

Question: {query}

Please answer the question based on the provided context. Cite sources using [Source N] notation."""


def build_summarization_prompt(text: str, max_length: int = 200) -> str:
    """Build a prompt for text summarization.
    
    :param text: Text to summarize
    :param max_length: Target summary length in words
    :returns: Formatted prompt
    """
    return f"""Please summarize the following research paper excerpt in approximately {max_length} words:

{text}

Summary:"""


def build_comparison_prompt(papers: list[dict]) -> str:
    """Build a prompt for comparing multiple papers.
    
    :param papers: List of paper dictionaries with title, abstract, etc.
    :returns: Formatted prompt
    """
    paper_texts = []
    for i, paper in enumerate(papers, 1):
        paper_texts.append(
            f"Paper {i}: {paper.get('title', 'Unknown')}\n"
            f"Abstract: {paper.get('abstract', 'No abstract available')}\n"
        )
    
    papers_str = "\n---\n".join(paper_texts)
    
    return f"""Compare and contrast the following research papers:

{papers_str}

Please analyze:
1. Main research questions/problems addressed
2. Methodologies used
3. Key findings and contributions
4. Similarities and differences
5. How the works relate to or build upon each other"""


def format_search_results_context(search_results: list[dict], max_chunks: int = 5) -> str:
    """Format search results into context string for RAG.
    
    :param search_results: List of search result dictionaries
    :param max_chunks: Maximum number of chunks to include
    :returns: Formatted context string
    """
    context_parts = []
    
    for i, result in enumerate(search_results[:max_chunks], 1):
        chunk_data = result.get('chunk', {})
        arxiv_meta = chunk_data.get('arxiv_metadata', {})
        chunk_meta = chunk_data.get('chunk_metadata', {})
        
        arxiv_id = arxiv_meta.get('arxiv_id', 'N/A')
        title = arxiv_meta.get('title', 'N/A')
        section = chunk_meta.get('section_heading', 'N/A')
        text = chunk_data.get('chunk_text', '')
        score = result.get('score', 0)
        
        context_parts.append(
            f"[Source {i}] (Relevance Score: {score:.3f})\n"
            f"Paper: {title}\n"
            f"arXiv ID: {arxiv_id}\n"
            f"Section: {section}\n"
            f"Content:\n{text}\n"
        )
    
    return "\n---\n".join(context_parts)
