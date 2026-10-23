from app.models import Chunk
from app.pipeline.components.chunkers import LangChainRecursiveChunker, LlamaIndexSentenceChunker
from app.pipeline.components.retrievers import HybridRRFRetriever, RetrievalContext
from tests.conftest import SAMPLE


def check_offsets(text, drafts):
    assert len(drafts) > 1
    for d in drafts:
        assert text[d.start_char : d.end_char] == d.text


def test_llamaindex_chunk_offsets_point_into_source_text():
    text = SAMPLE.read_text()
    chunker = LlamaIndexSentenceChunker(LlamaIndexSentenceChunker.Params(chunk_size=128, chunk_overlap=16))
    check_offsets(text, chunker.chunk(text))


def test_langchain_chunk_offsets_point_into_source_text():
    text = SAMPLE.read_text()
    chunker = LangChainRecursiveChunker(LangChainRecursiveChunker.Params(chunk_size=500, chunk_overlap=50))
    check_offsets(text, chunker.chunk(text))


class FakeStore:
    """Returns a fixed vector ranking so fusion can be tested without Chroma or Ollama."""

    def __init__(self, ranking):
        self.ranking = ranking

    def search(self, collection, embedding, k, document_ids):
        return self.ranking[:k]


def test_hybrid_fusion_records_both_rankings():
    chunks = [
        Chunk(id=f"c{i}", document_id="d", index=i, text=t, start_char=None, end_char=None)
        for i, t in enumerate(["vacation days policy", "laptop refresh cycle", "parental leave weeks"])
    ]
    ctx = RetrievalContext(
        query="how many vacation days",
        query_embedding=[0.0],
        store=FakeStore([("c2", 0.9), ("c0", 0.8)]),
        collection="x",
        chunks=chunks,
        document_ids=["d"],
    )
    results = HybridRRFRetriever(HybridRRFRetriever.Params(top_k=3)).retrieve(ctx)

    # c0 ranks 2nd by vector and 1st by BM25, so fusion puts it first.
    assert [r.chunk.id for r in results] == ["c0", "c2"]
    top = results[0].scores
    assert top["vector_rank"] == 2 and top["bm25_rank"] == 1
    assert results[1].scores["bm25_rank"] is None
