from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class IIDCategoricalSource:
    vocab_size: int
    probs: np.ndarray

    @classmethod
    def from_zipf(cls, vocab_size: int, alpha: float = 1.1, temperature: float = 1.0):
        ranks = np.arange(1, vocab_size + 1)
        logits = -alpha * np.log(ranks) / max(temperature, 1e-8)
        p = np.exp(logits - logits.max())
        p /= p.sum()
        return cls(vocab_size=vocab_size, probs=p)

    @classmethod
    def from_random_softmax(cls, vocab_size: int, seed: int = 0, scale: float = 1.0):
        rng = np.random.default_rng(seed)
        logits = rng.normal(0.0, scale, size=vocab_size)
        p = np.exp(logits - logits.max())
        p /= p.sum()
        return cls(vocab_size=vocab_size, probs=p)

    def entropy(self) -> float:
        p = np.clip(self.probs, 1e-12, 1.0)
        return float(-(p * np.log(p)).sum())

    def sample_sequence(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.choice(self.vocab_size, size=n, p=self.probs)

    def conditional_probs(self, prefix: np.ndarray, t: int) -> np.ndarray:
        return self.probs

    def log_prob_sequence(self, seq: np.ndarray) -> float:
        p = np.clip(self.probs[seq], 1e-300, 1.0)
        return float(np.log(p).sum())
