#!/usr/bin/env python3
"""Evaluate Graph-OT Calibration (GOTC) vs baselines.

Default uses synthetic toy embeddings designed to induce both hubness and noisy pairings.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from gotc.gotc import gotc, knn_graph_from_scores, propagate_scores, sinkhorn_calibrate


@dataclass
class Metrics:
    r1_i2t: float
    r5_i2t: float
    r10_i2t: float
    r1_t2i: float
    r5_t2i: float
    r10_t2i: float
    hub_gini: float
    hub_skew: float


def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def make_toy_embeddings(
    n: int = 1000,
    d: int = 128,
    clusters: int = 25,
    hub_count: int = 20,
    hub_strength: float = 1.0,
    noise_rate: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate paired embeddings with hubness and pair noise.

    Pair noise is injected into target embeddings directly (not just labels), so
    post-processing must correct noisy correspondences from relations.
    """
    rng = np.random.default_rng(seed)

    centers = rng.normal(size=(clusters, d))
    centers = l2_normalize(centers)
    cid = rng.integers(0, clusters, size=n)

    z = centers[cid] + 0.25 * rng.normal(size=(n, d))
    img = z + 0.25 * rng.normal(size=(n, d))
    txt = z + 0.25 * rng.normal(size=(n, d))

    hub_dir = rng.normal(size=(d,))
    hub_dir /= np.linalg.norm(hub_dir) + 1e-12
    hub_ids = rng.choice(n, size=min(hub_count, n), replace=False)
    txt[hub_ids] += hub_strength * hub_dir[None, :]

    img = l2_normalize(img)
    txt = l2_normalize(txt)

    gt = np.arange(n)

    # Inject pair noise by shuffling a fraction of target embeddings.
    if noise_rate > 0:
        n_noisy = int(noise_rate * n)
        noisy_idx = rng.choice(n, size=n_noisy, replace=False)
        shuffled = rng.permutation(noisy_idx)
        txt[noisy_idx] = txt[shuffled]

    txt = l2_normalize(txt)
    return img, txt, gt


def recall_at_k(S: np.ndarray, gt: np.ndarray, k: int, query_to_target: bool = True) -> float:
    if query_to_target:
        rank_idx = np.argpartition(-S, kth=min(k - 1, S.shape[1] - 1), axis=1)[:, :k]
        hit = (rank_idx == gt[:, None]).any(axis=1)
    else:
        ST = S.T
        rank_idx = np.argpartition(-ST, kth=min(k - 1, ST.shape[1] - 1), axis=1)[:, :k]
        hit = (rank_idx == gt[:, None]).any(axis=1)
    return float(hit.mean())


def gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if np.all(x == 0):
        return 0.0
    x = np.sort(np.maximum(x, 0.0))
    n = x.size
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def skewness(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    mu = x.mean()
    std = x.std()
    if std < 1e-12:
        return 0.0
    return float(np.mean(((x - mu) / std) ** 3))


def compute_metrics(S: np.ndarray, gt: np.ndarray) -> Metrics:
    top1 = np.argmax(S, axis=1)
    counts = np.bincount(top1, minlength=S.shape[1])
    return Metrics(
        r1_i2t=recall_at_k(S, gt, 1, True),
        r5_i2t=recall_at_k(S, gt, 5, True),
        r10_i2t=recall_at_k(S, gt, 10, True),
        r1_t2i=recall_at_k(S, gt, 1, False),
        r5_t2i=recall_at_k(S, gt, 5, False),
        r10_t2i=recall_at_k(S, gt, 10, False),
        hub_gini=gini(counts),
        hub_skew=skewness(counts),
    )


def fmt(m: Metrics) -> str:
    return (
        f"{m.r1_i2t:.3f}\t{m.r5_i2t:.3f}\t{m.r10_i2t:.3f}\t"
        f"{m.r1_t2i:.3f}\t{m.r5_t2i:.3f}\t{m.r10_t2i:.3f}\t"
        f"{m.hub_gini:.3f}\t{m.hub_skew:.3f}"
    )


def evaluate_once(args: argparse.Namespace, noise_rate: float) -> Dict[str, Metrics]:
    img, txt, gt = make_toy_embeddings(
        n=args.n,
        d=args.d,
        clusters=args.clusters,
        hub_count=args.hub_count,
        hub_strength=args.hub_strength,
        noise_rate=noise_rate,
        seed=args.seed,
    )
    S = img @ txt.T

    S_ot = sinkhorn_calibrate(S, iters=args.sinkhorn_iters, eps=args.eps)

    A_base_q = knn_graph_from_scores(S, k=args.k, sym=True)
    S_prop_q = propagate_scores(S, A_base_q, steps=args.prop_steps, alpha=args.alpha)
    A_base_t = knn_graph_from_scores(S.T, k=args.k, sym=True)
    S_prop_t = propagate_scores(S.T, A_base_t, steps=args.prop_steps, alpha=args.alpha).T
    S_prop = 0.5 * (S_prop_q + S_prop_t)

    S_gotc = gotc(
        S,
        k=args.k,
        outer_iters=args.outer_iters,
        sinkhorn_iters=args.sinkhorn_iters,
        eps=args.eps,
        prop_steps=args.prop_steps,
        alpha=args.alpha,
    )

    return {
        "baseline": compute_metrics(S, gt),
        "ot_only": compute_metrics(S_ot, gt),
        "prop_only": compute_metrics(S_prop, gt),
        "gotc": compute_metrics(S_gotc, gt),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--clusters", type=int, default=25)
    p.add_argument("--hub_count", type=int, default=20)
    p.add_argument("--hub_strength", type=float, default=1.0)
    p.add_argument("--k", type=int, default=25)
    p.add_argument("--outer_iters", type=int, default=3)
    p.add_argument("--sinkhorn_iters", type=int, default=5)
    p.add_argument("--eps", type=float, default=0.1)
    p.add_argument("--prop_steps", type=int, default=2)
    p.add_argument("--alpha", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise_rate", type=float, default=None, help="Single noise rate run")
    p.add_argument("--noise_grid", type=float, nargs="*", default=[0.0, 0.1, 0.2, 0.4])
    args = p.parse_args()

    grid = [args.noise_rate] if args.noise_rate is not None else list(args.noise_grid)
    for nr in grid:
        print(f"\n=== noise_rate={nr:.2f} ===")
        res = evaluate_once(args, nr)
        print("method\tR1_i2t\tR5_i2t\tR10_i2t\tR1_t2i\tR5_t2i\tR10_t2i\tHubGini\tHubSkew")
        for name in ["baseline", "ot_only", "prop_only", "gotc"]:
            print(f"{name}\t{fmt(res[name])}")


if __name__ == "__main__":
    main()
