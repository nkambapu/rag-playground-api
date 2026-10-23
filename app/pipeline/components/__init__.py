# Importing the modules registers every implementation in the registry.
from app.pipeline.components import (  # noqa: F401
    chunkers,
    embedders,
    generators,
    loaders,
    query_transforms,
    rerankers,
    retrievers,
    vector_stores,
)
