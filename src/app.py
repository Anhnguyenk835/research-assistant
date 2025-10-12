"""FastAPI application setup."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import get_settings
from routers import rag_router
# from services.opensearch_service.instance import make_opensearch_client_fresh
from services.embedding_service.instance import make_embeddings_client
from services.llm_service.instance import make_groq_client

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

clients = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for the application."""
    # Startup
    logger.info("Starting up Research Assistant API...")
    settings = get_settings()
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"OpenSearch host: {settings.opensearch.host}")

    # initialize clients to verify connectivity
    try:
        logger.info("Initializing service clients...")
        clients["embeddings_client"] = make_embeddings_client(settings=settings)
        clients["groq_client"] = make_groq_client(settings=settings)
        logger.info("All service clients initialized successfully.")

    except Exception as e:
        logger.error(f"Error initializing clients: {e}")
        raise

    yield

    # Shutdown - cleanup clients
    logger.info("Shutting down Research Assistant server...")
    logger.info("Closing service clients...")

    try:
        if "embeddings_client" in clients:
            await clients["embeddings_client"].close()
            logger.info("Embeddings client closed.")
        if "groq_client" in clients:
            await clients["groq_client"].close()
            logger.info("Groq client closed.")
    except Exception as e:
        logger.error(f"Error closing clients: {e}")
        raise

    # Shutdown
    logger.info("Server shutdown complete.")


# Create FastAPI app
app = FastAPI(
    title="Research Assistant API",
    description=(
        "API for querying research papers using RAG (Retrieval-Augmented Generation). "
        "Combines hybrid search (BM25 + vector similarity) with LLM-powered answer generation."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.state.clients = clients

# CORS middleware
settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(rag_router.router)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Research Assistant API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/rag/health"
    }


@app.get("/health", tags=["Root"])
async def health():
    """Basic health check."""
    return {
        "status": "healthy",
        "service": "Research Assistant API"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
