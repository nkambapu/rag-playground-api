from pathlib import Path

import pytest
from fastapi.testclient import TestClient

SAMPLE = Path(__file__).parent.parent / "samples" / "northwind-handbook.md"


@pytest.fixture
def client(tmp_path, monkeypatch):
    """An app instance with its own empty data folder."""
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path))
    from app.pipeline.components.vector_stores import chroma_client
    from app.settings import get_settings

    get_settings.cache_clear()
    chroma_client.cache_clear()
    from app.main import create_app

    with TestClient(create_app()) as test_client:
        yield test_client
    get_settings.cache_clear()
    chroma_client.cache_clear()


def ollama_available() -> bool:
    import ollama

    try:
        names = {m.model for m in ollama.Client(timeout=3).list().models}
    except Exception:
        return False
    return any(n.startswith("nomic-embed-text") for n in names) and any(n.startswith("qwen3:8b") for n in names)


requires_ollama = pytest.mark.skipif(not ollama_available(), reason="needs Ollama with qwen3:8b and nomic-embed-text")
