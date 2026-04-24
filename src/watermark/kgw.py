from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils.randomness import stable_hash_int


@dataclass
class KGWWatermarker:
    vocab_size: int
    gamma: float = 0.5
    delta: float = 1.0
    key: int = 12345
    context_dependent: bool = True

    def _seed_for_position(self, prefix: np.ndarray, t: int) -> int:
        if not self.context_dependent:
            return stable_hash_int([self.key, t], key=self.key)
        ctx = prefix[:t].tolist() if t > 0 else []
        return stable_hash_int([self.key, t, *ctx], key=self.key)

    def green_mask(self, prefix: np.ndarray, t: int) -> np.ndarray:
        seed = self._seed_for_position(prefix, t)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(self.vocab_size)
        k = int(round(self.gamma * self.vocab_size))
        mask = np.zeros(self.vocab_size, dtype=bool)
        mask[perm[:k]] = True
        return mask

    def reweight(self, base_probs: np.ndarray, prefix: np.ndarray, t: int) -> np.ndarray:
        g = self.green_mask(prefix, t)
        w = np.exp(self.delta * g.astype(float))
        q = base_probs * w
        q /= q.sum()
        return q

    def sample_watermarked(self, source, n: int, rng: np.random.Generator) -> np.ndarray:
        seq = np.zeros(n, dtype=int)
        for t in range(n):
            p = source.conditional_probs(seq, t)
            q = self.reweight(p, seq, t)
            seq[t] = int(rng.choice(self.vocab_size, p=q))
        return seq

    def log_prob_under_q(self, source, seq: np.ndarray) -> float:
        lp = 0.0
        prefix = np.zeros_like(seq)
        for t in range(len(seq)):
            p = source.conditional_probs(prefix, t)
            q = self.reweight(p, prefix, t)
            lp += np.log(np.clip(q[seq[t]], 1e-300, 1.0))
            prefix[t] = seq[t]
        return float(lp)

    def green_hits(self, seq: np.ndarray) -> int:
        hits = 0
        prefix = np.zeros_like(seq)
        for t in range(len(seq)):
            g = self.green_mask(prefix, t)
            hits += int(g[seq[t]])
            prefix[t] = seq[t]
        return hits
