"""
RAG Document Assistant — FastAPI Backend
Entry point: loads vector store once at startup, then serves requests.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.query import router as query_router
from app.core.config import settings
from app.services.retrieval import load_vector_store
from app.utils.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load expensive resources once at startup, clean up on shutdown."""
    logger.info("Starting up — loading vector store...")
    load_vector_store()
    logger.info("Vector store ready. API is up.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="RAG Document Assistant",
    description="Ask questions about your documents and get grounded, cited answers.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router, prefix="/api/v1", tags=["query"])


@app.get("/health", tags=["health"])
def health_check():
    return {"status": "ok", "service": "rag-assistant"}
