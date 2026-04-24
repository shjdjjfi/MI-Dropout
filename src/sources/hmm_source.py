from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp


@dataclass
class HMMSource:
    vocab_size: int
    num_states: int
    init_state_probs: np.ndarray
    state_transition: np.ndarray
    emission: np.ndarray
    fast_conditionals: bool = True

    @classmethod
    def random(
        cls,
        vocab_size: int,
        num_states: int,
        seed: int = 0,
        trans_temp: float = 1.0,
        emission_temp: float = 1.0,
    ):
        rng = np.random.default_rng(seed)
        init = rng.random(num_states)
        init /= init.sum()
        trans_logits = rng.normal(size=(num_states, num_states)) / max(trans_temp, 1e-8)
        trans = np.exp(trans_logits - trans_logits.max(axis=1, keepdims=True))
        trans /= trans.sum(axis=1, keepdims=True)
        emit_logits = rng.normal(size=(num_states, vocab_size)) / max(emission_temp, 1e-8)
        emit = np.exp(emit_logits - emit_logits.max(axis=1, keepdims=True))
        emit /= emit.sum(axis=1, keepdims=True)
        return cls(vocab_size, num_states, init, trans, emit)

    def sample_sequence(self, n: int, rng: np.random.Generator) -> np.ndarray:
        states = np.zeros(n, dtype=int)
        obs = np.zeros(n, dtype=int)
        states[0] = int(rng.choice(self.num_states, p=self.init_state_probs))
        obs[0] = int(rng.choice(self.vocab_size, p=self.emission[states[0]]))
        for t in range(1, n):
            states[t] = int(rng.choice(self.num_states, p=self.state_transition[states[t - 1]]))
            obs[t] = int(rng.choice(self.vocab_size, p=self.emission[states[t]]))
        return obs

    def conditional_probs(self, prefix: np.ndarray, t: int) -> np.ndarray:
        if t == 0:
            return self.init_state_probs @ self.emission
        if self.fast_conditionals:
            # Fast approximate filtering based on latest observation only.
            o = prefix[t - 1]
            post = self.emission[:, o] * self.init_state_probs
            post = post / np.clip(post.sum(), 1e-300, None)
            pred = post @ self.state_transition
            pred = pred / np.clip(pred.sum(), 1e-300, None)
            return pred @ self.emission
        alpha = np.log(np.clip(self.init_state_probs, 1e-300, 1.0)) + np.log(
            np.clip(self.emission[:, prefix[0]], 1e-300, 1.0)
        )
        for i in range(1, t):
            alpha = logsumexp(alpha[:, None] + np.log(np.clip(self.state_transition, 1e-300, 1.0)), axis=0)
            alpha = alpha + np.log(np.clip(self.emission[:, prefix[i]], 1e-300, 1.0))
        pred = logsumexp(alpha[:, None] + np.log(np.clip(self.state_transition, 1e-300, 1.0)), axis=0)
        pred = np.exp(pred - logsumexp(pred))
        return pred @ self.emission

    def log_prob_sequence(self, seq: np.ndarray) -> float:
        alpha = np.log(np.clip(self.init_state_probs, 1e-300, 1.0)) + np.log(
            np.clip(self.emission[:, seq[0]], 1e-300, 1.0)
        )
        for t in range(1, len(seq)):
            alpha = logsumexp(alpha[:, None] + np.log(np.clip(self.state_transition, 1e-300, 1.0)), axis=0)
            alpha += np.log(np.clip(self.emission[:, seq[t]], 1e-300, 1.0))
        return float(logsumexp(alpha))
