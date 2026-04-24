from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from metrics.quality import global_nll_degradation, local_conditional_degradation


@dataclass
class AttackResult:
    attacked_seq: np.ndarray
    success: bool
    score_before: float
    score_after: float
    edits: int
    quality_degradation: float
    runtime_sec: float
    steps: int
    approximate: bool = False


def quality_ok(source, original, candidate, eps: float, quality_mode: str = "global", local_mode: str = "average") -> bool:
    if quality_mode == "global":
        return global_nll_degradation(source, original, candidate) <= eps + 1e-12
    d = local_conditional_degradation(source, original, candidate, mode=local_mode)
    return d <= eps + 1e-12


def attack_template(run_fn):
    def wrapper(*args, **kwargs):
        t0 = perf_counter()
        out = run_fn(*args, **kwargs)
        out.runtime_sec = perf_counter() - t0
        return out

    return wrapper
