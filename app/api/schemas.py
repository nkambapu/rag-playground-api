"""Response models. They define the OpenAPI spec that the UI generates its TypeScript types from."""

from typing import Any

from pydantic import BaseModel

from app.models import Chunk
from app.pipeline.config import ResolvedConfig, StepConfig, StepOverride
from app.pipeline.trace import StepTrace


class OllamaStatus(BaseModel):
    base_url: str
    reachable: bool
    models: list[str] = []
    error: str | None = None


class Health(BaseModel):
    status: str
    ollama: OllamaStatus


class ImplementationInfo(BaseModel):
    name: str
    label: str
    library: str
    description: str
    defaults: dict[str, Any]
    params_schema: dict[str, Any]


class StepInfo(BaseModel):
    step: str
    label: str
    ingestion: bool
    default_impl: str
    implementations: list[ImplementationInfo]


class ResolveResult(BaseModel):
    config: ResolvedConfig
    index_key: str
    overridden: dict[str, list[str]]


class Preset(BaseModel):
    name: str
    overrides: dict[str, StepOverride]
    updated_at: str


class IndexInfo(BaseModel):
    index_key: str
    chunk_count: int
    config: dict[str, StepConfig]  # the ingestion settings that built this index
    created_at: str


class Document(BaseModel):
    id: str
    filename: str
    size_bytes: int
    created_at: str
    indexes: list[IndexInfo]


class IndexResult(BaseModel):
    document_id: str
    index_key: str
    chunk_count: int
    trace: list[StepTrace]


class UploadResult(IndexResult):
    document: Document


class DocumentChunks(BaseModel):
    document_id: str
    index_key: str
    text: str
    chunks: list[Chunk]


class ResultSummary(BaseModel):
    chunk_id: str
    document_id: str
    chunk_index: int
    score: float
    scores: dict[str, float | int | None]
    preview: str


class Source(ResultSummary):
    number: int
    text: str


class QueryResult(BaseModel):
    run_id: str
    question: str
    answer: str
    sources: list[Source]
    document_ids: list[str]
    newly_indexed: list[str]
    index_key: str
    config: ResolvedConfig
    overridden: dict[str, list[str]]
    trace: list[StepTrace]
    total_ms: float


class RunSummary(BaseModel):
    id: str
    question: str
    created_at: str


class Run(QueryResult):
    created_at: str
