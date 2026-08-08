"""
Retrieval service — loads the ChromaDB vector store built by the notebook
and exposes a function to find the top-K most relevant chunks.
Compatible with chromadb >= 1.0.0
"""
import logging
import os
from typing import Optional

import chromadb
from chromadb import Collection
from sentence_transformers import SentenceTransformer

from app.core.config import settings

logger = logging.getLogger(__name__)

# Module-level singletons — loaded once at startup
_collection: Optional[Collection] = None
_embedder: Optional[SentenceTransformer] = None


def load_vector_store() -> None:
    """Load ChromaDB collection and the embedding model into memory."""
    global _collection, _embedder

    logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
    _embedder = SentenceTransformer(settings.EMBEDDING_MODEL)

    store_path = os.path.abspath(settings.VECTOR_STORE_PATH)
    if not os.path.isdir(store_path):
        raise FileNotFoundError(
            f"Vector store not found at '{store_path}'. "
            "Run the notebook first to build it."
        )

    logger.info(f"Connecting to ChromaDB at: {store_path}")
    # chromadb 1.x uses Settings for path configuration
    client = chromadb.PersistentClient(path=store_path)
    _collection = client.get_collection(settings.COLLECTION_NAME)
    logger.info(
        f"Collection '{settings.COLLECTION_NAME}' loaded — "
        f"{_collection.count()} chunks."
    )


def retrieve_chunks(question: str) -> list[dict]:
    """
    Embed the question and return the top-K most similar chunks.

    Returns a list of dicts:
        { "document": str, "source": str, "chunk_id": str }
    """
    if _collection is None or _embedder is None:
        raise RuntimeError("Vector store not loaded. Did startup complete?")

    embedding = _embedder.encode(question).tolist()
    results = _collection.query(
        query_embeddings=[embedding],
        n_results=settings.TOP_K,
        include=["documents", "metadatas"],
    )

    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({
            "document": doc,
            "source": meta.get("source", "unknown"),
            "chunk_id": meta.get("chunk_id", "?"),
        })
    return chunks
