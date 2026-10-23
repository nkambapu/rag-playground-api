from functools import lru_cache

from pydantic import Field

from app.models import ScoredChunk
from app.pipeline.base import Component, Params, register


class Reranker(Component):
    step = "reranker"

    def rerank(self, question: str, results: list[ScoredChunk]) -> list[ScoredChunk]:
        raise NotImplementedError


@register
class NoReranker(Reranker):
    name = "none"
    label = "None"
    library = "built-in"
    description = "Keep the retriever's order and all of its results."

    def rerank(self, question: str, results: list[ScoredChunk]) -> list[ScoredChunk]:
        return results


@lru_cache(maxsize=2)
def load_cross_encoder(model: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model)


@register
class CrossEncoderReranker(Reranker):
    name = "sentence_transformers.cross_encoder"
    label = "Cross-encoder reranker"
    library = "sentence-transformers"
    description = "Scores each (question, chunk) pair together; more accurate than embeddings alone, but slower."

    class Params(Params):
        model: str = Field(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            description="Hugging Face cross-encoder model (e.g. BAAI/bge-reranker-base for higher quality)",
        )
        top_n: int = Field(3, ge=1, le=50, description="Chunks kept after reranking")

    def rerank(self, question: str, results: list[ScoredChunk]) -> list[ScoredChunk]:
        if not results:
            return []
        scores = load_cross_encoder(self.params.model).predict([(question, r.chunk.text) for r in results])
        for retrieval_rank, (r, s) in enumerate(zip(results, scores), start=1):
            r.scores["retrieval_rank"] = retrieval_rank
            r.scores["rerank_score"] = float(s)
            r.score = float(s)
        return sorted(results, key=lambda r: r.score, reverse=True)[: self.params.top_n]
