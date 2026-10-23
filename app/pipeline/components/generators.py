from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.models import ScoredChunk
from app.pipeline.base import Component, Params, register
from app.settings import get_settings

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using only the provided context. "
    "If the context does not contain the answer, say that you don't know. "
    "Cite the sources you used as [1], [2], and so on."
)
DEFAULT_PROMPT_TEMPLATE = "Context:\n{context}\n\nQuestion: {question}"


class Generation(BaseModel):
    answer: str
    messages: list[dict[str, str]]  # the exact prompt sent to the model
    model: str
    reasoning: str | None = None
    usage: dict[str, Any] = {}


class Generator(Component):
    step = "generator"

    def generate(self, question: str, sources: list[ScoredChunk]) -> Generation:
        raise NotImplementedError


def build_context(sources: list[ScoredChunk]) -> str:
    return "\n\n".join(f"[{i}] {s.chunk.text}" for i, s in enumerate(sources, start=1))


@register
class LangChainOllamaGenerator(Generator):
    name = "langchain.ollama"
    label = "Ollama chat model (LangChain)"
    library = "langchain"
    description = "Answers with a local Ollama model, using the retrieved chunks as numbered sources."

    class Params(Params):
        model: str = Field(default_factory=lambda: get_settings().default_llm_model, description="Ollama chat model")
        temperature: float = Field(0.1, ge=0, le=2, description="Higher is more creative, lower is more literal")
        num_ctx: int = Field(8192, ge=512, le=131072, description="Context window in tokens")
        think: bool = Field(False, description="Let reasoning models (e.g. qwen3) think before answering")
        system_prompt: str = Field(DEFAULT_SYSTEM_PROMPT, description="Instructions for the model")
        prompt_template: str = Field(DEFAULT_PROMPT_TEMPLATE, description="Must contain {context} and {question}")

        @field_validator("prompt_template")
        @classmethod
        def has_placeholders(cls, value: str) -> str:
            missing = [p for p in ("{context}", "{question}") if p not in value]
            if missing:
                raise ValueError(f"missing placeholder(s): {', '.join(missing)}")
            return value

    def generate(self, question: str, sources: list[ScoredChunk]) -> Generation:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_ollama import ChatOllama

        p = self.params
        user_prompt = p.prompt_template.replace("{context}", build_context(sources)).replace("{question}", question)
        llm = ChatOllama(
            model=p.model,
            base_url=get_settings().ollama_base_url,
            temperature=p.temperature,
            num_ctx=p.num_ctx,
            reasoning=p.think,
        )
        response = llm.invoke([SystemMessage(p.system_prompt), HumanMessage(user_prompt)])
        return Generation(
            answer=response.text,
            messages=[{"role": "system", "content": p.system_prompt}, {"role": "user", "content": user_prompt}],
            model=p.model,
            reasoning=response.additional_kwargs.get("reasoning_content"),
            usage=dict(response.usage_metadata or {}),
        )
