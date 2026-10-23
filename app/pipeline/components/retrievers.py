"""Retrievers. Hybrid fusion is done here (not with LangChain's EnsembleRetriever) so every
chunk keeps its vector score, BM25 score and ranks for the trace view."""

import re
from dataclasses import dataclass
from functools import cached_property

from pydantic import Field

from app.models import Chunk, ScoredChunk
from app.pipeline.base import Component, Params, register
from app.pipeline.components.vector_stores import ChromaVectorStore


@dataclass
class RetrievalContext:
    query: str
    query_embedding: list[float]
    store: ChromaVectorStore
    collection: str
    chunks: list[Chunk]  # every chunk in the documents being searched (used by BM25)
    document_ids: list[str]

    @cached_property
    def chunks_by_id(self) -> dict[str, Chunk]:
        return {c.id: c for c in self.chunks}


class Retriever(Component):
    step = "retriever"

    def retrieve(self, ctx: RetrievalContext) -> list[ScoredChunk]:
        raise NotImplementedError


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def vector_ranking(ctx: RetrievalContext, k: int) -> list[tuple[Chunk, float]]:
    hits = ctx.store.search(ctx.collection, ctx.query_embedding, k, ctx.document_ids)
    return [(ctx.chunks_by_id[i], sim) for i, sim in hits if i in ctx.chunks_by_id]


def bm25_ranking(ctx: RetrievalContext, k: int) -> list[tuple[Chunk, float]]:
    from rank_bm25 import BM25Okapi

    if not ctx.chunks:
        return []
    bm25 = BM25Okapi([tokenize(c.text) for c in ctx.chunks])
    scores = bm25.get_scores(tokenize(ctx.query))
    ranked = sorted(zip(ctx.chunks, scores), key=lambda pair: pair[1], reverse=True)
    return [(c, float(s)) for c, s in ranked[:k] if s > 0]


class TopKParams(Params):
    top_k: int = Field(5, ge=1, le=50, description="Number of chunks to return")


@register
class VectorRetriever(Retriever):
    name = "vector"
    label = "Vector search"
    library = "chromadb"
    description = "Nearest neighbors by embedding similarity."
    Params = TopKParams

    def retrieve(self, ctx: RetrievalContext) -> list[ScoredChunk]:
        return [
            ScoredChunk(chunk=c, score=sim, scores={"vector_similarity": sim, "vector_rank": rank})
            for rank, (c, sim) in enumerate(vector_ranking(ctx, self.params.top_k), start=1)
        ]


@register
class BM25Retriever(Retriever):
    name = "bm25"
    label = "BM25 keyword search"
    library = "rank-bm25"
    description = "Classic keyword relevance. Good for exact terms, names and codes."
    Params = TopKParams

    def retrieve(self, ctx: RetrievalContext) -> list[ScoredChunk]:
        return [
            ScoredChunk(chunk=c, score=s, scores={"bm25_score": s, "bm25_rank": rank})
            for rank, (c, s) in enumerate(bm25_ranking(ctx, self.params.top_k), start=1)
        ]


@register
class HybridRRFRetriever(Retriever):
    name = "hybrid_rrf"
    label = "Hybrid (vector + BM25)"
    library = "chromadb + rank-bm25"
    description = "Runs both searches and fuses the rankings with weighted Reciprocal Rank Fusion."

    class Params(TopKParams):
        candidate_k: int = Field(20, ge=1, le=200, description="Candidates taken from each search before fusing")
        vector_weight: float = Field(0.5, ge=0, le=1, description="Weight of vector search; BM25 gets 1 - this")
        rrf_k: int = Field(60, ge=1, le=1000, description="RRF constant; higher values flatten rank differences")

    def retrieve(self, ctx: RetrievalContext) -> list[ScoredChunk]:
        p = self.params
        fused: dict[str, ScoredChunk] = {}

        def add(ranking, weight, score_key, rank_key):
            for rank, (chunk, score) in enumerate(ranking, start=1):
                entry = fused.setdefault(
                    chunk.id,
                    ScoredChunk(
                        chunk=chunk,
                        score=0.0,
                        scores={"vector_similarity": None, "vector_rank": None, "bm25_score": None, "bm25_rank": None},
                    ),
                )
                entry.score += weight / (p.rrf_k + rank)
                entry.scores[score_key] = score
                entry.scores[rank_key] = rank

        add(vector_ranking(ctx, p.candidate_k), p.vector_weight, "vector_similarity", "vector_rank")
        add(bm25_ranking(ctx, p.candidate_k), 1 - p.vector_weight, "bm25_score", "bm25_rank")

        results = sorted(fused.values(), key=lambda s: s.score, reverse=True)[: p.top_k]
        for r in results:
            r.scores["rrf_score"] = r.score
        return results
