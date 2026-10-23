"""Library-neutral data types shared by every pipeline step.

LangChain and LlamaIndex objects are converted to these at the edges, so the rest of
the pipeline does not depend on which library produced them.
"""

from typing import Any

from pydantic import BaseModel


class LoadedDocument(BaseModel):
    text: str
    metadata: dict[str, Any] = {}


class ChunkDraft(BaseModel):
    """A chunk before it is stored: text plus its position in the loaded document text."""

    text: str
    start_char: int | None = None
    end_char: int | None = None
    metadata: dict[str, Any] = {}


class Chunk(BaseModel):
    id: str
    document_id: str
    index: int
    text: str
    start_char: int | None
    end_char: int | None
    metadata: dict[str, Any] = {}


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
    # Every score that contributed, e.g. vector similarity, BM25, fused and rerank scores.
    scores: dict[str, float | int | None] = {}
