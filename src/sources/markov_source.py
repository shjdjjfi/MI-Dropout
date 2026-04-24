from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FirstOrderMarkovSource:
    vocab_size: int
    init_probs: np.ndarray
    transition: np.ndarray
    regime: str = "medium"

    @classmethod
    def random(
        cls,
        vocab_size: int,
        seed: int = 0,
        sparsity: float = 0.2,
        temperature: float = 1.0,
        regime: str = "medium",
    ):
        rng = np.random.default_rng(seed)
        raw = rng.normal(0, 1, size=(vocab_size, vocab_size)) / max(temperature, 1e-8)
        mask = rng.random(size=(vocab_size, vocab_size)) > sparsity
        raw = np.where(mask, raw, -20.0)
        transition = np.exp(raw - raw.max(axis=1, keepdims=True))
        transition /= transition.sum(axis=1, keepdims=True)
        init = rng.random(vocab_size)
        init /= init.sum()
        if regime == "easy":
            transition = 0.8 * np.eye(vocab_size) + 0.2 * transition
            transition /= transition.sum(axis=1, keepdims=True)
        elif regime == "hard":
            transition = 0.2 * np.eye(vocab_size) + 0.8 * transition
            transition /= transition.sum(axis=1, keepdims=True)
        return cls(vocab_size=vocab_size, init_probs=init, transition=transition, regime=regime)

    def sample_sequence(self, n: int, rng: np.random.Generator) -> np.ndarray:
        x = np.zeros(n, dtype=int)
        x[0] = int(rng.choice(self.vocab_size, p=self.init_probs))
        for t in range(1, n):
            x[t] = int(rng.choice(self.vocab_size, p=self.transition[x[t - 1]]))
        return x

    def conditional_probs(self, prefix: np.ndarray, t: int) -> np.ndarray:
        if t == 0:
            return self.init_probs
        return self.transition[prefix[t - 1]]

    def log_prob_sequence(self, seq: np.ndarray) -> float:
        lp = np.log(np.clip(self.init_probs[seq[0]], 1e-300, 1.0))
        for t in range(1, len(seq)):
            lp += np.log(np.clip(self.transition[seq[t - 1], seq[t]], 1e-300, 1.0))
        return float(lp)

    def entropy_rate_estimate(self) -> float:
        vals, vecs = np.linalg.eig(self.transition.T)
        idx = int(np.argmin(np.abs(vals - 1.0)))
        stat = np.real(vecs[:, idx])
        stat = np.abs(stat)
        stat /= stat.sum()
        with np.errstate(divide="ignore"):
            h = -(self.transition * np.log(np.clip(self.transition, 1e-300, 1.0))).sum(axis=1)
        return float(np.dot(stat, h))
