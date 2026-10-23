from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import app.pipeline  # noqa: F401  (registers all implementations)
from app.api.routes import router
from app.pipeline.config import ConfigError
from app.pipeline.runner import Pipeline, PipelineError
from app.settings import get_settings
from app.store import Store


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    app.state.store = Store(settings.data_dir / "rag.db")
    app.state.pipeline = Pipeline(app.state.store)
    yield


def pipeline_error_response(exc: PipelineError) -> JSONResponse:
    cause = exc.__cause__
    if isinstance(cause, (httpx.ConnectError, ConnectionError)):
        code, message = 503, f"Cannot reach Ollama at {get_settings().ollama_base_url}. Is it running? ({cause})"
    else:
        code, message = 502, str(exc)
    return JSONResponse(status_code=code, content={"detail": message, "trace": exc.trace})


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="RAG Playground API",
        version="0.1.0",
        description="A configurable, fully traced RAG pipeline (LangChain + LlamaIndex + Ollama).",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_origin_regex=settings.cors_origin_regex,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)

    @app.exception_handler(ConfigError)
    async def config_error(_: Request, exc: ConfigError):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.exception_handler(PipelineError)
    async def pipeline_error(_: Request, exc: PipelineError):
        return pipeline_error_response(exc)

    return app


app = create_app()
