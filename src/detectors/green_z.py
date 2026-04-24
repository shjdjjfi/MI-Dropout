from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class GreenZDetector:
    gamma: float

    def score(self, seq, watermarker) -> dict:
        n = len(seq)
        hits = watermarker.green_hits(seq)
        mean = n * self.gamma
        var = max(n * self.gamma * (1 - self.gamma), 1e-8)
        z = (hits - mean) / np.sqrt(var)
        p_value = 1 - norm.cdf(z)
        return {"z": float(z), "p_value": float(p_value), "hits": int(hits)}
