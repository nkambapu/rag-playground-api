from pydantic import Field, model_validator

from app.models import ChunkDraft
from app.pipeline.base import Component, Params, register


class Chunker(Component):
    step = "chunker"

    def chunk(self, text: str) -> list[ChunkDraft]:
        raise NotImplementedError


class SizeOverlapParams(Params):
    @model_validator(mode="after")
    def overlap_smaller_than_size(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self


def fill_missing_offsets(text: str, drafts: list[ChunkDraft]) -> list[ChunkDraft]:
    """Some splitters don't report positions; find each chunk in the source text, in order."""
    cursor = 0
    for draft in drafts:
        if draft.start_char is None:
            found = text.find(draft.text, cursor)
            if found == -1:
                continue
            draft.start_char = found
        if draft.end_char is None:
            draft.end_char = draft.start_char + len(draft.text)
        cursor = draft.start_char + 1
    return drafts


@register
class LlamaIndexSentenceChunker(Chunker):
    name = "llamaindex.sentence"
    label = "LlamaIndex SentenceSplitter"
    library = "llama-index"
    description = "Splits on sentence boundaries, packing sentences up to a token budget."

    class Params(SizeOverlapParams):
        chunk_size: int = Field(256, ge=32, le=4096, description="Maximum chunk size in tokens")
        chunk_overlap: int = Field(32, ge=0, le=1024, description="Tokens shared between neighboring chunks")

    def chunk(self, text: str) -> list[ChunkDraft]:
        from llama_index.core import Document
        from llama_index.core.node_parser import SentenceSplitter

        splitter = SentenceSplitter(chunk_size=self.params.chunk_size, chunk_overlap=self.params.chunk_overlap)
        nodes = splitter.get_nodes_from_documents([Document(text=text)])
        drafts = [ChunkDraft(text=n.text, start_char=n.start_char_idx, end_char=n.end_char_idx) for n in nodes]
        return fill_missing_offsets(text, drafts)


@register
class LangChainRecursiveChunker(Chunker):
    name = "langchain.recursive"
    label = "LangChain RecursiveCharacterTextSplitter"
    library = "langchain"
    description = "Splits on paragraphs, then lines, then words, until chunks fit a character budget."

    class Params(SizeOverlapParams):
        chunk_size: int = Field(1000, ge=100, le=20000, description="Maximum chunk size in characters")
        chunk_overlap: int = Field(150, ge=0, le=5000, description="Characters shared between neighboring chunks")

    def chunk(self, text: str) -> list[ChunkDraft]:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.params.chunk_size, chunk_overlap=self.params.chunk_overlap, add_start_index=True
        )
        drafts = [
            ChunkDraft(text=d.page_content, start_char=d.metadata.get("start_index"))
            for d in splitter.create_documents([text])
        ]
        return fill_missing_offsets(text, drafts)
