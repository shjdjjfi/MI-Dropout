from __future__ import annotations

import numpy as np


def roc_auc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    y = np.concatenate([np.ones_like(pos_scores), np.zeros_like(neg_scores)])
    s = np.concatenate([pos_scores, neg_scores])
    order = np.argsort(s)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(s))
    r_pos = ranks[y == 1].sum()
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    auc = (r_pos - n_pos * (n_pos - 1) / 2) / max(n_pos * n_neg, 1)
    return float(auc)


def error_rates(pos_scores: np.ndarray, neg_scores: np.ndarray, alpha: float = 0.05) -> dict:
    threshold = np.quantile(neg_scores, 1 - alpha)
    type1 = float((neg_scores > threshold).mean())
    type2 = float((pos_scores <= threshold).mean())
    return {"threshold": float(threshold), "type1": type1, "type2": type2, "power": 1 - type2}
