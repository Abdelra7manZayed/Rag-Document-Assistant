"""
Query routes: GET /health, POST /query
"""
import logging

from fastapi import APIRouter, HTTPException

from app.core.config import settings
from app.schemas.query import QueryRequest, QueryResponse, SourceChunk
from app.services.generation import generate_answer
from app.services.retrieval import retrieve_chunks

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    """
    Retrieve relevant document chunks and generate a grounded answer.
    """
    logger.info(f"Query received: {request.question!r}")

    try:
        chunks = retrieve_chunks(request.question)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    if not chunks:
        return QueryResponse(
            answer="No relevant documents found for your question.",
            sources=[],
            model_used=settings.OLLAMA_MODEL,
        )

    try:
        answer = generate_answer(request.question, chunks)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    sources = [
        SourceChunk(
            source=c["source"],
            chunk_id=c["chunk_id"],
            excerpt=c["document"][:200] + "...",
        )
        for c in chunks
    ]

    return QueryResponse(
        answer=answer,
        sources=sources,
        model_used=settings.OLLAMA_MODEL,
    )
