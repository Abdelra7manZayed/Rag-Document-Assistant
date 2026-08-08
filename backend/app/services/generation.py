"""
Generation service — builds the RAG prompt from retrieved chunks
and calls the Ollama LLM to produce a grounded, cited answer.
"""
import logging

import ollama

from app.core.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful document assistant.
Answer the user's question using ONLY the context provided below.
Each context chunk is labelled [Source: <filename> | Chunk: <id>].
At the end of your answer, list the chunk IDs you used as citations like:
  Sources used: chunk_1, chunk_3

If the context does not contain enough information to answer, say:
"I could not find a clear answer in the provided documents."
Do NOT use your own general knowledge — ground every claim in the context."""


def build_prompt(question: str, chunks: list[dict]) -> str:
    """Combine retrieved chunks into a single context block."""
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        context_parts.append(
            f"[Source: {chunk['source']} | Chunk: {chunk['chunk_id']}]\n"
            f"{chunk['document']}"
        )
    context = "\n\n---\n\n".join(context_parts)
    return f"Context:\n{context}\n\nQuestion: {question}"


def generate_answer(question: str, chunks: list[dict]) -> str:
    """Call Ollama and return the model's response."""
    prompt = build_prompt(question, chunks)

    logger.info(f"Calling Ollama model '{settings.OLLAMA_MODEL}'...")
    try:
        response = ollama.chat(
            model=settings.OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            options={"temperature": 0.1},   # low temp → more faithful retrieval
        )
        return response["message"]["content"]
    except Exception as e:
        logger.error(f"Ollama call failed: {e}")
        raise RuntimeError(
            f"LLM call failed. Is Ollama running at {settings.OLLAMA_BASE_URL}? "
            f"Error: {e}"
        )
