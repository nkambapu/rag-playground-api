import json
import re
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Form, HTTPException, Query, Request, UploadFile, status
from pydantic import BaseModel, Field, ValidationError

from app.pipeline.base import INGESTION_STEPS, REGISTRY, STEPS
from app.pipeline.components.vector_stores import delete_document_vectors
from app.pipeline.config import DEFAULT_IMPLS, ConfigError, ConfigRequest, ResolvedConfig, StepOverride, resolve
from app.pipeline.runner import Pipeline
from app.settings import get_settings
from app.store import Store, new_id

router = APIRouter(prefix="/api/v1")

ALLOWED_SUFFIXES = {".pdf", ".txt", ".md"}
PRESET_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _-]{0,63}$")


def get_store(request: Request) -> Store:
    return request.app.state.store


def get_pipeline(request: Request) -> Pipeline:
    return request.app.state.pipeline


StoreDep = Annotated[Store, Depends(get_store)]
PipelineDep = Annotated[Pipeline, Depends(get_pipeline)]


def resolve_config(store: Store, request: ConfigRequest | None) -> ResolvedConfig:
    preset_overrides = None
    if request and request.preset and request.preset != "default":
        preset = store.get_preset(request.preset)
        if preset is None:
            raise ConfigError(f"Preset {request.preset!r} not found")
        preset_overrides = {k: StepOverride(**v) for k, v in preset["overrides"].items()}
    return resolve(request, preset_overrides)


def require_document(store: Store, document_id: str) -> dict[str, Any]:
    doc = store.get_document(document_id)
    if doc is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Document {document_id} not found")
    return doc


# Health and pipeline options


@router.get("/health")
def health() -> dict[str, Any]:
    import ollama

    settings = get_settings()
    try:
        models = sorted(m.model for m in ollama.Client(host=settings.ollama_base_url, timeout=3).list().models)
        ollama_status = {"reachable": True, "models": models}
    except Exception as exc:
        ollama_status = {"reachable": False, "error": str(exc)}
    return {"status": "ok", "ollama": {"base_url": settings.ollama_base_url, **ollama_status}}


@router.get("/pipeline/options")
def pipeline_options() -> list[dict[str, Any]]:
    """Every step with its implementations, default settings and JSON schema for the settings form."""
    return [
        {
            "step": step,
            "label": label,
            "ingestion": step in INGESTION_STEPS,
            "default_impl": DEFAULT_IMPLS[step],
            "implementations": [
                {
                    "name": cls.name,
                    "label": cls.label,
                    "library": cls.library,
                    "description": cls.description,
                    "defaults": cls.Params().model_dump(),
                    "params_schema": cls.Params.model_json_schema(),
                }
                for cls in REGISTRY[step].values()
            ],
        }
        for step, label in STEPS.items()
    ]


@router.post("/pipeline/resolve")
def resolve_pipeline(store: StoreDep, body: ConfigRequest) -> dict[str, Any]:
    """Validates a config and returns the full settings, the index key and what differs from defaults."""
    config = resolve_config(store, body)
    return {"config": config.model_dump(), "index_key": config.index_key, "overridden": config.overridden()}


# Presets


class PresetBody(BaseModel):
    overrides: dict[str, StepOverride] = {}


@router.get("/presets")
def list_presets(store: StoreDep) -> list[dict[str, Any]]:
    return store.list_presets()


@router.put("/presets/{name}")
def save_preset(store: StoreDep, name: str, body: PresetBody) -> dict[str, Any]:
    if name == "default" or not PRESET_NAME.match(name):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "Preset names use letters, numbers, spaces, - and _ ('default' is reserved)"
        )
    resolve(ConfigRequest(overrides=body.overrides))  # validate before saving
    return store.save_preset(name, {k: v.model_dump(exclude_none=True) for k, v in body.overrides.items()})


@router.delete("/presets/{name}", status_code=status.HTTP_204_NO_CONTENT)
def delete_preset(store: StoreDep, name: str) -> None:
    if not store.delete_preset(name):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Preset {name!r} not found")


# Documents


@router.get("/documents")
def list_documents(store: StoreDep) -> list[dict[str, Any]]:
    return store.list_documents()


@router.post("/documents", status_code=status.HTTP_201_CREATED)
async def upload_document(
    store: StoreDep,
    pipeline: PipelineDep,
    file: UploadFile,
    config: Annotated[str | None, Form(description="Optional ConfigRequest as JSON")] = None,
) -> dict[str, Any]:
    """Uploads a file and indexes it with the given configuration (defaults if omitted)."""
    settings = get_settings()
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, f"Supported file types: {', '.join(sorted(ALLOWED_SUFFIXES))}")
    try:
        config_request = ConfigRequest(**json.loads(config)) if config else None
    except (json.JSONDecodeError, TypeError, ValidationError) as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, f"config must be a ConfigRequest JSON object: {exc}")
    resolved = resolve_config(store, config_request)

    content = await file.read()
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"Files are limited to {settings.max_upload_mb} MB")

    doc_id = new_id()
    path = settings.data_dir / "uploads" / f"{doc_id}{suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    store.add_document(doc_id, Path(file.filename).name, path, len(content))
    try:
        result = pipeline.index(doc_id, resolved)
    except Exception:
        store.delete_document(doc_id)
        delete_document_vectors(doc_id)
        path.unlink(missing_ok=True)
        raise
    return {"document": store.get_document(doc_id), **result}


@router.get("/documents/{document_id}")
def get_document(store: StoreDep, document_id: str) -> dict[str, Any]:
    return require_document(store, document_id)


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_document(store: StoreDep, document_id: str) -> None:
    doc = require_document(store, document_id)
    delete_document_vectors(document_id)
    store.delete_document(document_id)
    Path(doc["path"]).unlink(missing_ok=True)


@router.post("/documents/{document_id}/index")
def index_document(store: StoreDep, pipeline: PipelineDep, document_id: str, body: ConfigRequest) -> dict[str, Any]:
    """(Re)indexes a document with the given configuration."""
    require_document(store, document_id)
    return pipeline.index(document_id, resolve_config(store, body))


@router.get("/documents/{document_id}/chunks")
def document_chunks(
    store: StoreDep,
    document_id: str,
    index_key: Annotated[str | None, Query(description="Defaults to the index built with default settings")] = None,
) -> dict[str, Any]:
    """The indexed text and its chunks with character offsets, for the chunk viewer."""
    require_document(store, document_id)
    key = index_key or resolve(None).index_key
    text = store.get_indexed_text(key, document_id)
    if text is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Document {document_id} has no index {key}")
    chunks = store.get_chunks(key, [document_id])
    return {"document_id": document_id, "index_key": key, "text": text, "chunks": [c.model_dump() for c in chunks]}


# Querying and run history


class QueryBody(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    document_ids: list[str] | None = Field(None, description="Documents to search; all documents if omitted")
    config: ConfigRequest = ConfigRequest()


@router.post("/query")
def query(store: StoreDep, pipeline: PipelineDep, body: QueryBody) -> dict[str, Any]:
    config = resolve_config(store, body.config)
    document_ids = body.document_ids or [d["id"] for d in store.list_documents()]
    if not document_ids:
        raise HTTPException(status.HTTP_409_CONFLICT, "Upload a document before asking questions")
    for doc_id in document_ids:
        require_document(store, doc_id)
    return pipeline.query(body.question.strip(), config, document_ids)


@router.get("/runs")
def list_runs(store: StoreDep, limit: Annotated[int, Query(ge=1, le=200)] = 50) -> list[dict[str, Any]]:
    return store.list_runs(limit)


@router.get("/runs/{run_id}")
def get_run(store: StoreDep, run_id: str) -> dict[str, Any]:
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Run {run_id} not found")
    return run
