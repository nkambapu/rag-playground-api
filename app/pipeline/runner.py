"""Runs the pipeline: ingestion (load -> chunk -> embed -> store) and querying
(transform -> retrieve -> rerank -> generate), recording a trace for every step."""

from pathlib import Path
from typing import Any

from app.models import Chunk, ScoredChunk
from app.pipeline.base import Component, get_component_class
from app.pipeline.config import ResolvedConfig
from app.pipeline.components.retrievers import RetrievalContext
from app.pipeline.trace import Tracer
from app.store import Store, new_id

PREVIEW_CHARS = 160


def build(config: ResolvedConfig, step: str) -> Any:
    cfg = config.steps[step]
    cls: type[Component] = get_component_class(step, cfg.impl)
    return cls(cls.Params(**cfg.params))


def preview(text: str) -> str:
    return text if len(text) <= PREVIEW_CHARS else text[:PREVIEW_CHARS] + "…"


def summarize(results: list[ScoredChunk]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": r.chunk.id,
            "document_id": r.chunk.document_id,
            "chunk_index": r.chunk.index,
            "score": r.score,
            "scores": dict(r.scores),
            "preview": preview(r.chunk.text),
        }
        for r in results
    ]


class PipelineError(Exception):
    """A step failed. Carries the trace up to and including the failed step."""

    def __init__(self, message: str, trace: list[dict[str, Any]]):
        super().__init__(message)
        self.trace = trace


class Pipeline:
    def __init__(self, store: Store):
        self.store = store

    def ingest(self, document_id: str, config: ResolvedConfig, tracer: Tracer) -> int:
        doc = self.store.get_document(document_id)
        if doc is None:
            raise LookupError(f"Document {document_id} not found")
        index_key = config.index_key
        steps = config.steps

        with tracer.step("loader", steps["loader"]) as out:
            loaded = build(config, "loader").load(Path(doc["path"]))
            out.update(document_id=document_id, characters=len(loaded.text), metadata=loaded.metadata)

        with tracer.step("chunker", steps["chunker"]) as out:
            drafts = build(config, "chunker").chunk(loaded.text)
            chunks = [
                Chunk(
                    id=f"{index_key}-{document_id}-{i}",
                    document_id=document_id,
                    index=i,
                    text=d.text,
                    start_char=d.start_char,
                    end_char=d.end_char,
                    metadata=d.metadata,
                )
                for i, d in enumerate(drafts)
            ]
            lengths = [len(c.text) for c in chunks]
            out.update(
                document_id=document_id,
                chunk_count=len(chunks),
                avg_chars=round(sum(lengths) / len(lengths), 1) if lengths else 0,
                min_chars=min(lengths, default=0),
                max_chars=max(lengths, default=0),
            )

        with tracer.step("embedder", steps["embedder"]) as out:
            embeddings = build(config, "embedder").embed_documents([c.text for c in chunks]) if chunks else []
            out.update(document_id=document_id, vectors=len(embeddings), dimensions=len(embeddings[0]) if embeddings else 0)

        with tracer.step("vector_store", steps["vector_store"]) as out:
            build(config, "vector_store").add(index_key, chunks, embeddings)
            self.store.save_index(
                index_key, document_id, loaded.text, chunks,
                {s: steps[s].model_dump() for s in ("loader", "chunker", "embedder", "vector_store")},
            )
            out.update(document_id=document_id, collection=index_key, stored=len(chunks))

        return len(chunks)

    def ensure_indexed(self, document_ids: list[str], config: ResolvedConfig, tracer: Tracer) -> list[str]:
        """Indexes any documents that have no index for this configuration yet. Returns their ids."""
        missing = [d for d in document_ids if not self.store.is_indexed(config.index_key, d)]
        for doc_id in missing:
            self.ingest(doc_id, config, tracer)
        return missing

    def query(self, question: str, config: ResolvedConfig, document_ids: list[str]) -> dict[str, Any]:
        tracer = Tracer()
        try:
            return self._query(question, config, document_ids, tracer)
        except Exception as exc:
            raise PipelineError(str(exc), [t.model_dump() for t in tracer.steps]) from exc

    def index(self, document_id: str, config: ResolvedConfig) -> dict[str, Any]:
        tracer = Tracer()
        try:
            chunk_count = self.ingest(document_id, config, tracer)
        except Exception as exc:
            raise PipelineError(str(exc), [t.model_dump() for t in tracer.steps]) from exc
        return {
            "document_id": document_id,
            "index_key": config.index_key,
            "chunk_count": chunk_count,
            "trace": [t.model_dump() for t in tracer.steps],
        }

    def _query(self, question: str, config: ResolvedConfig, document_ids: list[str], tracer: Tracer) -> dict[str, Any]:
        steps = config.steps
        newly_indexed = self.ensure_indexed(document_ids, config, tracer)
        embedder = build(config, "embedder")
        store = build(config, "vector_store")
        all_chunks = self.store.get_chunks(config.index_key, document_ids)

        with tracer.step("query_transform", steps["query_transform"]) as out:
            queries = build(config, "query_transform").transform(question)
            out.update(queries=queries)

        with tracer.step("retriever", steps["retriever"]) as out:
            retriever = build(config, "retriever")
            merged: dict[str, ScoredChunk] = {}
            for q in queries:
                ctx = RetrievalContext(
                    query=q,
                    query_embedding=embedder.embed_query(q),
                    store=store,
                    collection=config.index_key,
                    chunks=all_chunks,
                    document_ids=document_ids,
                )
                for r in retriever.retrieve(ctx):
                    if r.chunk.id not in merged or r.score > merged[r.chunk.id].score:
                        merged[r.chunk.id] = r
            retrieved = sorted(merged.values(), key=lambda r: r.score, reverse=True)
            out.update(searched_chunks=len(all_chunks), results=summarize(retrieved))

        with tracer.step("reranker", steps["reranker"]) as out:
            before = [r.chunk.id for r in retrieved]
            sources = build(config, "reranker").rerank(question, [r.model_copy(deep=True) for r in retrieved])
            kept = {r.chunk.id for r in sources}
            out.update(results=summarize(sources), dropped=[cid for cid in before if cid not in kept])

        with tracer.step("generator", steps["generator"]) as out:
            generation = build(config, "generator").generate(question, sources)
            out.update(
                model=generation.model,
                messages=generation.messages,
                answer=generation.answer,
                reasoning=generation.reasoning,
                usage=generation.usage,
            )

        result = {
            "run_id": new_id(),
            "question": question,
            "answer": generation.answer,
            "sources": [
                {"number": i, **summary, "text": src.chunk.text}
                for i, (summary, src) in enumerate(zip(summarize(sources), sources), start=1)
            ],
            "document_ids": document_ids,
            "newly_indexed": newly_indexed,
            "index_key": config.index_key,
            "config": config.model_dump(),
            "overridden": config.overridden(),
            "trace": [t.model_dump() for t in tracer.steps],
            "total_ms": round(sum(t.duration_ms for t in tracer.steps), 2),
        }
        self.store.save_run(result["run_id"], question, result)
        return result
