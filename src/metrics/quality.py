from __future__ import annotations

import numpy as np


def global_nll(source, seq: np.ndarray) -> float:
    return float(-source.log_prob_sequence(seq) / len(seq))


def global_nll_degradation(source, original: np.ndarray, attacked: np.ndarray) -> float:
    return global_nll(source, attacked) - global_nll(source, original)


def local_conditional_degradation(source, original: np.ndarray, attacked: np.ndarray, mode: str = "average") -> float:
    assert len(original) == len(attacked)
    deltas = []
    pref_o = np.zeros_like(original)
    pref_a = np.zeros_like(attacked)
    for t in range(len(original)):
        if original[t] != attacked[t]:
            p_o = source.conditional_probs(pref_o, t)
            p_a = source.conditional_probs(pref_a, t)
            d = np.log(np.clip(p_o[original[t]], 1e-300, 1.0)) - np.log(np.clip(p_a[attacked[t]], 1e-300, 1.0))
            deltas.append(float(d))
        pref_o[t] = original[t]
        pref_a[t] = attacked[t]
    if not deltas:
        return 0.0
    if mode == "hard":
        return float(np.max(deltas))
    return float(np.mean(deltas))
