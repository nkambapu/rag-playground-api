import pytest

import app.pipeline  # noqa: F401
from app.pipeline.config import ConfigError, ConfigRequest, StepOverride, default_config, resolve


def overrides(**steps) -> ConfigRequest:
    return ConfigRequest(overrides={k: StepOverride(**v) for k, v in steps.items()})


def test_defaults_use_each_library_where_it_is_strongest():
    steps = default_config().steps
    assert steps["loader"].impl.startswith("llamaindex.")
    assert steps["chunker"].impl == "llamaindex.sentence"
    assert steps["embedder"].impl == "langchain.ollama"
    assert steps["generator"].impl == "langchain.ollama"
    assert steps["chunker"].params == {"chunk_size": 256, "chunk_overlap": 32}


def test_param_override_keeps_other_params():
    config = resolve(overrides(chunker={"params": {"chunk_size": 512}}))
    assert config.steps["chunker"].params == {"chunk_size": 512, "chunk_overlap": 32}
    assert config.overridden() == {"chunker": ["chunk_size"]}


def test_switching_impl_starts_from_its_own_defaults():
    config = resolve(overrides(chunker={"impl": "langchain.recursive", "params": {"chunk_overlap": 100}}))
    assert config.steps["chunker"].params == {"chunk_size": 1000, "chunk_overlap": 100}
    assert config.overridden() == {"chunker": ["impl"]}


def test_preset_then_request_overrides():
    preset = {"retriever": StepOverride(params={"top_k": 8, "vector_weight": 0.7})}
    config = resolve(overrides(retriever={"params": {"top_k": 3}}), preset)
    assert config.steps["retriever"].params["top_k"] == 3
    assert config.steps["retriever"].params["vector_weight"] == 0.7


def test_ingestion_changes_change_the_index_key_but_query_settings_do_not():
    base = default_config().index_key
    assert resolve(overrides(chunker={"params": {"chunk_size": 512}})).index_key != base
    assert resolve(overrides(retriever={"params": {"top_k": 9}})).index_key == base


@pytest.mark.parametrize(
    "request_overrides, message",
    [
        ({"chunker": {"params": {"chunk_size": 100, "chunk_overlap": 100}}}, "chunk_overlap must be smaller"),
        ({"chunker": {"params": {"chunk_sise": 100}}}, "chunk_sise"),
        ({"chunker": {"impl": "magic"}}, "Unknown implementation 'magic'"),
        ({"chunking": {"impl": "langchain.recursive"}}, "Unknown step 'chunking'"),
        ({"generator": {"params": {"prompt_template": "{question}"}}}, "{context}"),
    ],
)
def test_invalid_overrides_are_rejected(request_overrides, message):
    with pytest.raises(ConfigError, match=message.replace("{", r"\{").replace("}", r"\}")):
        resolve(overrides(**request_overrides))
