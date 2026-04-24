from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class SyntheticSource(Protocol):
    vocab_size: int

    def sample_sequence(self, n: int, rng: np.random.Generator) -> np.ndarray: ...

    def conditional_probs(self, prefix: np.ndarray, t: int) -> np.ndarray: ...

    def log_prob_sequence(self, seq: np.ndarray) -> float: ...


@dataclass
class SourceConfig:
    vocab_size: int
    seed: int = 0
