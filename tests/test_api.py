import json

from tests.conftest import SAMPLE, requires_ollama


def test_options_list_every_step_with_defaults(client):
    options = client.get("/api/v1/pipeline/options").json()
    assert [o["step"] for o in options] == [
        "loader", "chunker", "embedder", "vector_store", "query_transform", "retriever", "reranker", "generator",
    ]
    chunker = next(o for o in options if o["step"] == "chunker")
    assert chunker["default_impl"] == "llamaindex.sentence"
    assert {i["name"] for i in chunker["implementations"]} >= {"llamaindex.sentence", "langchain.recursive"}
    assert "chunk_size" in chunker["implementations"][0]["params_schema"]["properties"]


def test_resolve_reports_errors_and_overrides(client):
    bad = client.post("/api/v1/pipeline/resolve", json={"overrides": {"retriever": {"params": {"top_k": 0}}}})
    assert bad.status_code == 422
    assert "top_k" in bad.json()["detail"]

    ok = client.post("/api/v1/pipeline/resolve", json={"overrides": {"retriever": {"params": {"top_k": 7}}}})
    assert ok.json()["overridden"] == {"retriever": ["top_k"]}


def test_presets_round_trip(client):
    body = {"overrides": {"reranker": {"impl": "none"}}}
    assert client.put("/api/v1/presets/no-rerank", json=body).status_code == 200
    resolved = client.post("/api/v1/pipeline/resolve", json={"preset": "no-rerank"}).json()
    assert resolved["config"]["steps"]["reranker"]["impl"] == "none"
    assert client.put("/api/v1/presets/default", json=body).status_code == 422
    assert client.delete("/api/v1/presets/no-rerank").status_code == 204


def test_rejects_unsupported_file_types(client):
    response = client.post("/api/v1/documents", files={"file": ("notes.docx", b"x")})
    assert response.status_code == 415


@requires_ollama
def test_upload_index_and_query_end_to_end(client):
    with SAMPLE.open("rb") as f:
        upload = client.post("/api/v1/documents", files={"file": (SAMPLE.name, f, "text/markdown")})
    assert upload.status_code == 201, upload.text
    doc_id = upload.json()["document"]["id"]
    assert [t["step"] for t in upload.json()["trace"]] == ["loader", "chunker", "embedder", "vector_store"]

    chunks = client.get(f"/api/v1/documents/{doc_id}/chunks").json()
    assert len(chunks["chunks"]) == upload.json()["chunk_count"] > 1

    result = client.post("/api/v1/query", json={"question": "How many paid vacation days do employees get?"}).json()
    assert [t["step"] for t in result["trace"]] == ["query_transform", "retriever", "reranker", "generator"]
    assert "22" in result["answer"]
    assert any("22 days" in s["text"] for s in result["sources"])
    assert client.get(f"/api/v1/runs/{result['run_id']}").status_code == 200

    # A different chunker needs a new index, which the query builds automatically.
    other = {"overrides": {"chunker": {"impl": "langchain.recursive"}}, "preset": None}
    result2 = client.post("/api/v1/query", json={"question": "Who founded the company?", "config": other}).json()
    assert result2["newly_indexed"] == [doc_id]
    assert result2["index_key"] != result["index_key"]
    assert json.dumps(result2["trace"]).count('"step": "chunker"') == 1
