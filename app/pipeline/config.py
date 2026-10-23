"""Layered pipeline configuration: built-in defaults -> saved preset -> per-request overrides.

Overrides only list what changes. Switching a step's implementation starts from that
implementation's own defaults, then applies the override params on top.
"""

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ValidationError

from app.pipeline.base import INGESTION_STEPS, STEPS, get_component_class

DEFAULT_IMPLS: dict[str, str] = {
    "loader": "llamaindex.simple_directory_reader",
    "chunker": "llamaindex.sentence",
    "embedder": "langchain.ollama",
    "vector_store": "chroma",
    "query_transform": "none",
    "retriever": "hybrid_rrf",
    "reranker": "sentence_transformers.cross_encoder",
    "generator": "langchain.ollama",
}


class ConfigError(ValueError):
    pass


class StepConfig(BaseModel):
    impl: str
    params: dict[str, Any] = {}


class StepOverride(BaseModel):
    impl: str | None = None
    params: dict[str, Any] = {}


class ConfigRequest(BaseModel):
    """What clients send: an optional preset name plus overrides for individual steps."""

    preset: str | None = None
    overrides: dict[str, StepOverride] = {}


class ResolvedConfig(BaseModel):
    steps: dict[str, StepConfig]

    @property
    def index_key(self) -> str:
        """Identifies the stored index: documents must be re-indexed when ingestion settings change."""
        ingestion = {step: self.steps[step].model_dump() for step in INGESTION_STEPS}
        digest = hashlib.sha256(json.dumps(ingestion, sort_keys=True).encode()).hexdigest()
        return f"idx_{digest[:12]}"

    def overridden(self) -> dict[str, list[str]]:
        """For each step that differs from the defaults: which fields changed ("impl" or param names)."""
        defaults = default_config().steps
        changed: dict[str, list[str]] = {}
        for step, cfg in self.steps.items():
            base = defaults[step]
            if cfg.impl != base.impl:
                changed[step] = ["impl"]
            else:
                keys = [k for k, v in cfg.params.items() if base.params.get(k) != v]
                if keys:
                    changed[step] = keys
        return changed


def _validated_params(step: str, impl: str, params: dict[str, Any]) -> dict[str, Any]:
    try:
        cls = get_component_class(step, impl)
    except LookupError as exc:
        raise ConfigError(str(exc)) from exc
    try:
        return cls.Params(**params).model_dump()
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'params'}: {e['msg']}" for e in exc.errors())
        raise ConfigError(f"Invalid settings for {step} ({impl}): {problems}") from exc


def default_config() -> ResolvedConfig:
    return ResolvedConfig(
        steps={step: StepConfig(impl=impl, params=_validated_params(step, impl, {})) for step, impl in DEFAULT_IMPLS.items()}
    )


def apply_overrides(config: ResolvedConfig, overrides: dict[str, StepOverride]) -> ResolvedConfig:
    steps = dict(config.steps)
    for step, override in overrides.items():
        if step not in STEPS:
            raise ConfigError(f"Unknown step {step!r} (steps: {', '.join(STEPS)})")
        current = steps[step]
        impl = override.impl or current.impl
        base_params = current.params if impl == current.impl else {}
        steps[step] = StepConfig(impl=impl, params=_validated_params(step, impl, {**base_params, **override.params}))
    return ResolvedConfig(steps=steps)


def resolve(request: ConfigRequest | None, preset_overrides: dict[str, StepOverride] | None = None) -> ResolvedConfig:
    config = default_config()
    if preset_overrides:
        config = apply_overrides(config, preset_overrides)
    if request and request.overrides:
        config = apply_overrides(config, request.overrides)
    return config
