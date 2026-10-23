# RAG Playground API

The backend for **RAG Playground**, a tool that makes every step of a Retrieval-Augmented Generation pipeline visible and configurable. Each request returns a full trace (what every step received, produced, scored and how long it took), so a UI can show *why* an answer came out the way it did.

Runs entirely on local, open-source models through [Ollama](https://ollama.com).

Frontend: [rag-playground-ui](https://github.com/nkambapu/rag-playground-ui) (React + TypeScript).

## Pipeline

```
Ingestion:  load → chunk → embed → store
Query:      transform query → retrieve → rerank → generate
```

Each step has a default implementation, chosen for what each library does best, and every step can be overridden per request.

| Step | Default | Other options |
|---|---|---|
| Loading | LlamaIndex `SimpleDirectoryReader` | LangChain loaders |
| Chunking | LlamaIndex `SentenceSplitter` (256 tokens, 32 overlap) | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | Ollama `nomic-embed-text` via LangChain | any Ollama embedding model |
| Vector store | Chroma (cosine) | L2 or inner-product distance |
| Query transformation | none | *(planned: multi-query, HyDE)* |
| Retrieval | Hybrid: vector + BM25, fused with weighted Reciprocal Rank Fusion | vector only, BM25 only |
| Reranking | Cross-encoder (`ms-marco-MiniLM-L-6-v2`), keeps top 3 | none, or any Hugging Face cross-encoder |
| Generation | Ollama `qwen3:8b` via LangChain | any Ollama chat model, custom prompts |

Hybrid fusion is implemented directly rather than with LangChain's `EnsembleRetriever`, so every chunk keeps its vector similarity, BM25 score and both ranks for the trace.

## Configuration

Settings are layered: **built-in defaults → saved preset → per-request overrides**. Overrides list only what changes:

```json
{
  "preset": "default",
  "overrides": {
    "chunker": { "impl": "langchain.recursive", "params": { "chunk_size": 800 } },
    "retriever": { "params": { "top_k": 8, "vector_weight": 0.7 } }
  }
}
```

- Switching an implementation starts from that implementation's own defaults.
- Every setting is validated (unknown keys, ranges, `chunk_overlap < chunk_size`, prompt placeholders) and errors name the step and field.
- Ingestion settings (loader, chunker, embedder, vector store) determine an **index key**. A query whose index doesn't exist yet indexes the documents automatically, and the trace includes those steps.
- `GET /api/v1/pipeline/options` describes every step, implementation, default and JSON schema, so the UI can build its settings form without hard-coding them.

## Getting started

Requirements: Python 3.12+, [uv](https://docs.astral.sh/uv/) and Ollama.

1. Pull the default models:

   ```bash
   ollama pull qwen3:8b
   ```

   ```bash
   ollama pull nomic-embed-text
   ```

2. Install dependencies:

   ```bash
   uv sync
   ```

3. Start the API:

   ```bash
   uv run uvicorn app.main:app --reload --port 8000
   ```

4. Open the interactive docs at http://localhost:8000/docs and try it with the sample document:

   ```bash
   curl -F "file=@samples/northwind-handbook.md" http://localhost:8000/api/v1/documents
   ```

   ```bash
   curl -X POST http://localhost:8000/api/v1/query -H "content-type: application/json" -d '{"question": "Can I carry over unused vacation days?"}'
   ```

The first query downloads the reranker model (about 90 MB) from Hugging Face.

Settings can be changed with environment variables or a `.env` file; see [.env.example](.env.example). Requests from any `localhost` port are allowed (for the UI's dev server); add other UI origins with `RAG_CORS_ORIGINS`.

All responses use typed Pydantic models, so the OpenAPI spec at `/openapi.json` fully describes them. The UI generates its TypeScript types from it.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/health` | API status, Ollama reachability and installed models |
| GET | `/api/v1/pipeline/options` | Steps, implementations, defaults and settings schemas |
| POST | `/api/v1/pipeline/resolve` | Validate a config; returns full settings, index key and what's overridden |
| GET / PUT / DELETE | `/api/v1/presets[/{name}]` | Saved named configurations |
| GET / POST | `/api/v1/documents` | List, or upload and index (`.pdf`, `.md`, `.txt`) |
| GET / DELETE | `/api/v1/documents/{id}` | Get or delete a document |
| POST | `/api/v1/documents/{id}/index` | Re-index with a given config |
| GET | `/api/v1/documents/{id}/chunks` | Indexed text plus chunks with character offsets (for highlighting) |
| POST | `/api/v1/query` | Ask a question; returns the answer, sources and full trace |
| GET | `/api/v1/runs[/{id}]` | Run history |

When a step fails, the error response includes the trace up to the failing step.

## Project layout

```
app/
├── main.py                 # FastAPI app, CORS, error handling
├── settings.py             # environment settings
├── models.py               # library-neutral Chunk / ScoredChunk types
├── store.py                # SQLite: documents, chunks, presets, runs
├── api/routes.py           # HTTP endpoints
└── pipeline/
    ├── base.py             # Component base class and registry
    ├── config.py           # defaults, layering, validation, index key
    ├── trace.py            # per-step trace recording
    ├── runner.py           # runs ingestion and queries
    └── components/         # one module per step; each implementation registers itself
samples/                    # sample document for demos and tests
tests/
```

To add an implementation, subclass the step's base class in `app/pipeline/components/`, give it a `Params` model and decorate it with `@register`. It then shows up in `/pipeline/options` and can be selected in any config.

## Tests

```bash
uv run pytest
```

The end-to-end test runs only when Ollama is reachable with `qwen3:8b` and `nomic-embed-text` installed; the other tests need no models.

## Roadmap

- More implementations: semantic chunking (LangChain and LlamaIndex), multi-query and HyDE query transformation, LlamaIndex retrievers and generators
- Streaming traces with Server-Sent Events
- Side-by-side comparison of two configurations
- Evaluation with RAGAS (Ollama as judge)
- Docker Compose setup
