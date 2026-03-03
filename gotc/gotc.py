"""Graph-OT Calibration (GOTC) for cross-modal retrieval."""

from __future__ import annotations

import numpy as np
from scipy import sparse

ArrayLike = np.ndarray


def _as_prob(v: ArrayLike | None, n: int) -> ArrayLike:
    if v is None:
        v = np.ones(n, dtype=np.float64) / float(n)
    v = np.asarray(v, dtype=np.float64)
    if v.ndim != 1 or v.shape[0] != n:
        raise ValueError(f"Expected vector of length {n}, got shape {v.shape}.")
    s = float(v.sum())
    if s <= 0:
        raise ValueError("Marginal mass must be positive.")
    return v / s


def sinkhorn_calibrate(
    S: ArrayLike,
    r: ArrayLike | None = None,
    c: ArrayLike | None = None,
    iters: int = 20,
    eps: float = 0.05,
) -> ArrayLike:
    """Calibrate similarity matrix with entropic OT Sinkhorn iterations."""
    S = np.asarray(S, dtype=np.float64)
    if S.ndim != 2:
        raise ValueError(f"S must be 2D, got {S.ndim}D")
    if eps <= 0:
        raise ValueError("eps must be positive")

    n, m = S.shape
    r = _as_prob(r, n)
    c = _as_prob(c, m)

    logits = S / eps
    logits -= logits.max()
    K = np.exp(logits)
    K = np.maximum(K, 1e-300)

    u = np.ones(n, dtype=np.float64)
    v = np.ones(m, dtype=np.float64)
    for _ in range(max(1, iters)):
        u = r / np.maximum(K @ v, 1e-300)
        v = c / np.maximum(K.T @ u, 1e-300)

    P = (u[:, None] * K) * v[None, :]
    return np.log(np.maximum(P, 1e-300))


def _normalize_rows(A: sparse.csr_matrix) -> sparse.csr_matrix:
    deg = np.asarray(A.sum(axis=1)).ravel()
    return (sparse.diags(1.0 / np.maximum(deg, 1e-12)) @ A).tocsr()


def knn_graph_from_scores(S_ot: ArrayLike, k: int = 20, sym: bool = True) -> sparse.csr_matrix:
    """Build row-normalized query graph from score profiles (CSR adjacency)."""
    X = np.asarray(S_ot, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("S_ot must be 2D")

    n = X.shape[0]
    if n == 0:
        return sparse.csr_matrix((0, 0), dtype=np.float64)

    k = int(max(1, min(k, n - 1 if n > 1 else 1)))

    X = X - X.mean(axis=1, keepdims=True)
    Xn = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    sim = Xn @ Xn.T
    np.fill_diagonal(sim, -np.inf)

    idx = np.argpartition(-sim, kth=min(k - 1, n - 1), axis=1)[:, :k]

    rows = np.repeat(np.arange(n), k)
    cols = idx.reshape(-1)
    vals = np.maximum(sim[np.arange(n)[:, None], idx].reshape(-1), 0.0)

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64)
    if sym:
        A = A.maximum(A.T)
    A = A + sparse.eye(n, dtype=np.float64, format="csr")
    return _normalize_rows(A)


def propagate_scores(S_ot: ArrayLike, A: sparse.csr_matrix, steps: int = 2, alpha: float = 0.9) -> ArrayLike:
    """Iterative propagation: S_{t+1}=alpha*A*S_t+(1-alpha)*S0."""
    if not sparse.isspmatrix_csr(A):
        A = A.tocsr()

    S0 = np.asarray(S_ot, dtype=np.float64)
    S = S0.copy()
    for _ in range(max(1, steps)):
        S_next = alpha * (A @ S) + (1.0 - alpha) * S0
        if np.linalg.norm(S_next - S) <= 1e-6 * np.linalg.norm(S):
            S = S_next
            break
        S = S_next
    return S


def _top2_margin(S: ArrayLike) -> ArrayLike:
    part = np.partition(S, kth=-2, axis=1)
    m = (part[:, -1] - part[:, -2]).reshape(-1, 1)
    return m


def _confidence_fusion(S_ot: ArrayLike, S_q: ArrayLike, S_t: ArrayLike) -> ArrayLike:
    """Conservative uncertainty-aware fusion."""
    S_ref = np.asarray(S_ot, dtype=np.float64)
    S_avg = 0.5 * (np.asarray(S_q, dtype=np.float64) + np.asarray(S_t, dtype=np.float64))

    margin_q = _top2_margin(S_ref)
    margin_t = _top2_margin(S_ref.T).T
    m = 0.5 * (margin_q + margin_t)
    m = (m - m.min()) / np.maximum(m.max() - m.min(), 1e-12)

    # only uncertain rows get noticeable propagation blending
    w = 0.02 + 0.23 * (1.0 - m)
    return (1.0 - w) * S_ref + w * S_avg


def _adaptive_target_marginal(S: ArrayLike, tau: float = 0.7) -> ArrayLike:
    top1 = np.argmax(S, axis=1)
    counts = np.bincount(top1, minlength=S.shape[1]).astype(np.float64)
    inv = 1.0 / np.power(np.maximum(counts, 1.0), tau)
    inv /= np.maximum(inv.sum(), 1e-12)
    return inv


def gotc(
    S: ArrayLike,
    k: int = 20,
    outer_iters: int = 2,
    sinkhorn_iters: int = 20,
    eps: float = 0.05,
    prop_steps: int = 2,
    alpha: float = 0.9,
) -> ArrayLike:
    """Alternate OT calibration and bidirectional relation propagation."""
    S_orig = np.asarray(S, dtype=np.float64)
    S_cur = S_orig.copy()
    eps_cur = float(eps)

    for _ in range(max(1, outer_iters)):
        c = _adaptive_target_marginal(S_cur, tau=0.4)
        S_ot = sinkhorn_calibrate(S_cur, c=c, iters=sinkhorn_iters, eps=max(eps_cur, 1e-3))

        A_q = knn_graph_from_scores(S_ot, k=k, sym=True)
        S_q = propagate_scores(S_ot, A_q, steps=prop_steps, alpha=alpha)

        A_t = knn_graph_from_scores(S_ot.T, k=k, sym=True)
        S_t = propagate_scores(S_ot.T, A_t, steps=prop_steps, alpha=alpha).T

        S_mix = _confidence_fusion(S_ot, S_q, S_t)
        # keep OT as backbone, use propagation for correction and retain raw similarity for precision
        S_cur = 0.60 * S_ot + 0.25 * S_mix + 0.15 * S_orig
        eps_cur *= 0.9  # anneal

    return S_cur
