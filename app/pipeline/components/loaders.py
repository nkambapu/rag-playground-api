from pathlib import Path

from app.models import LoadedDocument
from app.pipeline.base import Component, register

PAGE_SEPARATOR = "\n\n"


class Loader(Component):
    step = "loader"

    def load(self, path: Path) -> LoadedDocument:
        raise NotImplementedError


@register
class LlamaIndexLoader(Loader):
    name = "llamaindex.simple_directory_reader"
    label = "LlamaIndex SimpleDirectoryReader"
    library = "llama-index"
    description = "Picks a reader by file type (PDF, Markdown, text). Strong PDF handling."

    def load(self, path: Path) -> LoadedDocument:
        from llama_index.core import SimpleDirectoryReader

        docs = SimpleDirectoryReader(input_files=[str(path)]).load_data()
        parts = [d.text for d in docs if d.text.strip()]
        return LoadedDocument(text=PAGE_SEPARATOR.join(parts), metadata={"sections": len(parts)})


@register
class LangChainLoader(Loader):
    name = "langchain.loaders"
    label = "LangChain document loaders"
    library = "langchain"
    description = "PyPDFLoader for PDFs, TextLoader for text and Markdown (kept as raw text)."

    def load(self, path: Path) -> LoadedDocument:
        from langchain_community.document_loaders import PyPDFLoader, TextLoader

        loader = PyPDFLoader(str(path)) if path.suffix.lower() == ".pdf" else TextLoader(str(path), encoding="utf-8")
        parts = [d.page_content for d in loader.load() if d.page_content.strip()]
        return LoadedDocument(text=PAGE_SEPARATOR.join(parts), metadata={"sections": len(parts)})
