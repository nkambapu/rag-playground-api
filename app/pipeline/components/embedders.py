from pydantic import Field

from app.pipeline.base import Component, Params, register
from app.settings import get_settings


class Embedder(Component):
    step = "embedder"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError


@register
class LangChainOllamaEmbedder(Embedder):
    name = "langchain.ollama"
    label = "Ollama embeddings (LangChain)"
    library = "langchain"
    description = "Embeds text with a local Ollama embedding model."

    class Params(Params):
        model: str = Field(default_factory=lambda: get_settings().default_embed_model, description="Ollama embedding model")
        # nomic-embed-text is trained with these task prefixes; clear them for models that don't use them.
        document_prefix: str = Field("search_document: ", description="Text added before each chunk")
        query_prefix: str = Field("search_query: ", description="Text added before the question")

    def _client(self):
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(model=self.params.model, base_url=get_settings().ollama_base_url)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._client().embed_documents([self.params.document_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._client().embed_query(self.params.query_prefix + text)
