"""Per-step trace recording. The UI renders these records to show what each step did."""

import time
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel

from app.pipeline.config import StepConfig


class StepTrace(BaseModel):
    step: str
    impl: str
    params: dict[str, Any]
    started_at: str
    duration_ms: float
    status: Literal["ok", "error"]
    output: dict[str, Any] = {}
    error: str | None = None


class Tracer:
    def __init__(self) -> None:
        self.steps: list[StepTrace] = []

    @contextmanager
    def step(self, step: str, config: StepConfig):
        """Times a step and records it. The caller fills the yielded dict with the step's output."""
        output: dict[str, Any] = {}
        started_at = datetime.now(UTC).isoformat()
        start = time.perf_counter()
        status, error = "ok", None
        try:
            yield output
        except Exception as exc:
            status, error = "error", f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.steps.append(
                StepTrace(
                    step=step,
                    impl=config.impl,
                    params=config.params,
                    started_at=started_at,
                    duration_ms=round((time.perf_counter() - start) * 1000, 2),
                    status=status,
                    output=output,
                    error=error,
                )
            )
