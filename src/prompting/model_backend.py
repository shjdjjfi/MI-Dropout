from __future__ import annotations

import logging
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.95


class DummyBackend:
    """Deterministic backend for smoke tests/offline runs."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    def generate(self, prompt: str, cfg: GenerationConfig | None = None) -> str:
        cfg = cfg or GenerationConfig()
        first_line = prompt.strip().splitlines()[0][:50]
        return f"[{self.model_name}] {first_line} -> predicted_answer"
