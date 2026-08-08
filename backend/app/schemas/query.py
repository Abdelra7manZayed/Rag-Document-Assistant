"""
Request and response schemas for the /query endpoint.
"""
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        examples=["What is a Python generator?"],
    )


class SourceChunk(BaseModel):
    source: str           # document filename
    chunk_id: str         # unique chunk identifier
    excerpt: str          # short snippet of the chunk text


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    model_used: str
