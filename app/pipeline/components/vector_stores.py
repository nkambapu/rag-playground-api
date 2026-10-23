from functools import lru_cache
from typing import Literal

from pydantic import Field

from app.models import Chunk
from app.pipeline.base import Component, Params, register
from app.settings import get_settings


@lru_cache
def chroma_client():
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    path = get_settings().data_dir / "chroma"
    return chromadb.PersistentClient(path=str(path), settings=ChromaSettings(anonymized_telemetry=False))


class VectorStore(Component):
    step = "vector_store"


@register
class ChromaVectorStore(VectorStore):
    name = "chroma"
    label = "Chroma"
    library = "chromadb"
    description = "Local, persistent vector database. One collection per index configuration."

    class Params(Params):
        distance: Literal["cosine", "l2", "ip"] = Field("cosine", description="Distance metric for similarity search")

    def _collection(self, name: str):
        return chroma_client().get_or_create_collection(name, metadata={"hnsw:space": self.params.distance})

    def add(self, collection: str, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return
        self._collection(collection).add(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[{"document_id": c.document_id, "index": c.index} for c in chunks],
        )

    def search(
        self, collection: str, embedding: list[float], k: int, document_ids: list[str]
    ) -> list[tuple[str, float]]:
        """Returns (chunk id, similarity) pairs, best first. Similarity is derived from the distance."""
        where = {"document_id": {"$in": document_ids}} if len(document_ids) > 1 else {"document_id": document_ids[0]}
        result = self._collection(collection).query(query_embeddings=[embedding], n_results=k, where=where)
        pairs = zip(result["ids"][0], result["distances"][0])
        if self.params.distance == "l2":
            return [(i, 1 / (1 + d)) for i, d in pairs]
        return [(i, 1 - d) for i, d in pairs]  # Chroma's cosine and ip distances are 1 - similarity


def delete_document_vectors(document_id: str) -> None:
    for collection in chroma_client().list_collections():
        collection.delete(where={"document_id": document_id})
