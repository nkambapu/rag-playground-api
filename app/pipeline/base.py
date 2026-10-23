"""Component base class and the registry that maps (step, implementation name) to a class.

Every pipeline step has one or more implementations. Each implementation declares a
Pydantic `Params` model, which is used to validate overrides and to describe the
settings to the UI (via its JSON schema).
"""

from typing import ClassVar

from pydantic import BaseModel, ConfigDict


class Params(BaseModel):
    """Base for component settings. Unknown keys are rejected so typos surface as errors."""

    model_config = ConfigDict(extra="forbid")


class Component:
    step: ClassVar[str]
    name: ClassVar[str]
    label: ClassVar[str]
    library: ClassVar[str]
    description: ClassVar[str]
    Params: ClassVar[type[Params]] = Params

    def __init__(self, params: Params):
        self.params = params


# Pipeline steps in execution order. Ingestion steps decide what gets stored in the index.
STEPS: dict[str, str] = {
    "loader": "Loading / parsing",
    "chunker": "Chunking",
    "embedder": "Embeddings",
    "vector_store": "Vector store",
    "query_transform": "Query transformation",
    "retriever": "Retrieval",
    "reranker": "Reranking",
    "generator": "Generation",
}
INGESTION_STEPS = ("loader", "chunker", "embedder", "vector_store")

REGISTRY: dict[str, dict[str, type[Component]]] = {step: {} for step in STEPS}


def register[C: type[Component]](cls: C) -> C:
    if cls.step not in REGISTRY:
        raise ValueError(f"Unknown step {cls.step!r} for {cls.__name__}")
    REGISTRY[cls.step][cls.name] = cls
    return cls


def get_component_class(step: str, name: str) -> type[Component]:
    try:
        return REGISTRY[step][name]
    except KeyError:
        available = ", ".join(REGISTRY.get(step, {})) or "none"
        raise LookupError(f"Unknown implementation {name!r} for step {step!r} (available: {available})")
