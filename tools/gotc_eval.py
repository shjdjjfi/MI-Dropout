#!/usr/bin/env python3
"""Evaluate Graph-OT Calibration (GOTC) vs baselines with richer metrics."""

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
    map_i2t: float
    map_t2i: float
    ndcg10_i2t: float
    ndcg10_t2i: float
    cov1: float
    cov10: float
    top1_entropy: float
    hub_occ_1pct: float
    hub_gini: float
    hub_skew: float


def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def make_toy_embeddings(
    n: int = 1000,
    d: int = 128,
    clusters: int = 40,
    hub_count: int = 40,
    hub_strength: float = 1.2,
    noise_rate: float = 0.0,
    hard_swap_rate: float = 0.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    centers = l2_normalize(rng.normal(size=(clusters, d)))
    cid = rng.integers(0, clusters, size=n)

    z = centers[cid] + 0.45 * rng.normal(size=(n, d))
    img = 0.8 * z + 0.8 * rng.normal(size=(n, d))
    txt = 0.8 * z + 0.8 * rng.normal(size=(n, d))

    hub_dir = rng.normal(size=(d,))
    hub_dir /= np.linalg.norm(hub_dir) + 1e-12
    hub_ids = rng.choice(n, size=min(hub_count, n), replace=False)
    txt[hub_ids] += hub_strength * hub_dir[None, :]

    img = l2_normalize(img)
    txt = l2_normalize(txt)
    gt = np.arange(n)

    if noise_rate > 0:
        n_noisy = int(noise_rate * n)
        noisy_idx = rng.choice(n, size=n_noisy, replace=False)
        shuffled = rng.permutation(noisy_idx)
        txt[noisy_idx] = txt[shuffled]

    if hard_swap_rate > 0:
        n_hard = int(hard_swap_rate * n)
        chosen = rng.choice(n, size=n_hard, replace=False)
        for i in chosen:
            same = np.where(cid == cid[i])[0]
            if same.size > 1:
                j = int(rng.choice(same[same != i]))
                txt[i], txt[j] = txt[j].copy(), txt[i].copy()

    txt = l2_normalize(txt)
    return img, txt, gt


def recall_at_k(S: np.ndarray, gt: np.ndarray, k: int, query_to_target: bool = True) -> float:
    M = S if query_to_target else S.T
    rank_idx = np.argpartition(-M, kth=min(k - 1, M.shape[1] - 1), axis=1)[:, :k]
    hit = (rank_idx == gt[:, None]).any(axis=1)
    return float(hit.mean())


def _ranks_of_gt(S: np.ndarray, gt: np.ndarray, query_to_target: bool = True) -> np.ndarray:
    M = S if query_to_target else S.T
    order = np.argsort(-M, axis=1)
    # with gt = arange(n), lookup is straightforward
    ranks = np.argmax(order == gt[:, None], axis=1) + 1
    return ranks.astype(np.float64)


def map_single_positive(S: np.ndarray, gt: np.ndarray, query_to_target: bool = True) -> float:
    ranks = _ranks_of_gt(S, gt, query_to_target)
    return float(np.mean(1.0 / ranks))


def ndcg_at_k_single_positive(S: np.ndarray, gt: np.ndarray, k: int = 10, query_to_target: bool = True) -> float:
    ranks = _ranks_of_gt(S, gt, query_to_target)
    gains = np.where(ranks <= k, 1.0 / np.log2(ranks + 1.0), 0.0)
    return float(np.mean(gains))


def coverage_at_k(S: np.ndarray, k: int = 1) -> float:
    idx = np.argpartition(-S, kth=min(k - 1, S.shape[1] - 1), axis=1)[:, :k]
    return float(np.unique(idx).size / S.shape[1])


def top1_entropy_and_occ(counts: np.ndarray) -> tuple[float, float]:
    p = counts / np.maximum(counts.sum(), 1e-12)
    nz = p > 0
    ent = -np.sum(p[nz] * np.log(p[nz])) / np.log(len(p))
    topk = max(1, int(np.ceil(0.01 * len(counts))))
    occ = float(np.sort(counts)[-topk:].sum() / np.maximum(counts.sum(), 1e-12))
    return float(ent), occ


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
    ent, occ = top1_entropy_and_occ(counts)
    return Metrics(
        r1_i2t=recall_at_k(S, gt, 1, True),
        r5_i2t=recall_at_k(S, gt, 5, True),
        r10_i2t=recall_at_k(S, gt, 10, True),
        r1_t2i=recall_at_k(S, gt, 1, False),
        r5_t2i=recall_at_k(S, gt, 5, False),
        r10_t2i=recall_at_k(S, gt, 10, False),
        map_i2t=map_single_positive(S, gt, True),
        map_t2i=map_single_positive(S, gt, False),
        ndcg10_i2t=ndcg_at_k_single_positive(S, gt, 10, True),
        ndcg10_t2i=ndcg_at_k_single_positive(S, gt, 10, False),
        cov1=coverage_at_k(S, 1),
        cov10=coverage_at_k(S, 10),
        top1_entropy=ent,
        hub_occ_1pct=occ,
        hub_gini=gini(counts),
        hub_skew=skewness(counts),
    )


def fmt(m: Metrics) -> str:
    return (
        f"{m.r1_i2t:.3f}\t{m.r5_i2t:.3f}\t{m.r10_i2t:.3f}\t"
        f"{m.r1_t2i:.3f}\t{m.r5_t2i:.3f}\t{m.r10_t2i:.3f}\t"
        f"{m.map_i2t:.3f}\t{m.map_t2i:.3f}\t"
        f"{m.ndcg10_i2t:.3f}\t{m.ndcg10_t2i:.3f}\t"
        f"{m.cov1:.3f}\t{m.cov10:.3f}\t{m.top1_entropy:.3f}\t"
        f"{m.hub_occ_1pct:.3f}\t{m.hub_gini:.3f}\t{m.hub_skew:.3f}"
    )


def avg_metrics(ms: list[Metrics]) -> Metrics:
    vals = np.array([
        [
            m.r1_i2t, m.r5_i2t, m.r10_i2t, m.r1_t2i, m.r5_t2i, m.r10_t2i,
            m.map_i2t, m.map_t2i, m.ndcg10_i2t, m.ndcg10_t2i,
            m.cov1, m.cov10, m.top1_entropy, m.hub_occ_1pct, m.hub_gini, m.hub_skew,
        ]
        for m in ms
    ])
    mu = vals.mean(axis=0)
    return Metrics(*[float(x) for x in mu])


def evaluate_once(args: argparse.Namespace, noise_rate: float, seed: int) -> Dict[str, Metrics]:
    img, txt, gt = make_toy_embeddings(
        n=args.n,
        d=args.d,
        clusters=args.clusters,
        hub_count=args.hub_count,
        hub_strength=args.hub_strength,
        noise_rate=noise_rate,
        hard_swap_rate=args.hard_swap_rate,
        seed=seed,
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
    p.add_argument("--clusters", type=int, default=40)
    p.add_argument("--hub_count", type=int, default=40)
    p.add_argument("--hub_strength", type=float, default=1.2)
    p.add_argument("--hard_swap_rate", type=float, default=0.2)
    p.add_argument("--k", type=int, default=30)
    p.add_argument("--outer_iters", type=int, default=3)
    p.add_argument("--sinkhorn_iters", type=int, default=20)
    p.add_argument("--eps", type=float, default=0.06)
    p.add_argument("--prop_steps", type=int, default=2)
    p.add_argument("--alpha", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--num_seeds", type=int, default=5)
    p.add_argument("--noise_rate", type=float, default=None)
    p.add_argument("--noise_grid", type=float, nargs="*", default=[0.0, 0.1, 0.2, 0.4])
    args = p.parse_args()

    grid = [args.noise_rate] if args.noise_rate is not None else list(args.noise_grid)
    curve = {name: [] for name in ["baseline", "ot_only", "prop_only", "gotc"]}

    for nr in grid:
        agg: Dict[str, list[Metrics]] = {k: [] for k in ["baseline", "ot_only", "prop_only", "gotc"]}
        for sid in range(args.seed, args.seed + args.num_seeds):
            out = evaluate_once(args, nr, sid)
            for name in agg:
                agg[name].append(out[name])

        print(f"\n=== noise_rate={nr:.2f} (avg over {args.num_seeds} seeds) ===")
        print(
            "method\tR1_i2t\tR5_i2t\tR10_i2t\tR1_t2i\tR5_t2i\tR10_t2i\t"
            "mAP_i2t\tmAP_t2i\tnDCG10_i2t\tnDCG10_t2i\t"
            "Cov@1\tCov@10\tEntropy\tHubOcc1%\tHubGini\tHubSkew"
        )
        for name in ["baseline", "ot_only", "prop_only", "gotc"]:
            m = avg_metrics(agg[name])
            curve[name].append(m.r1_i2t)
            print(f"{name}\t{fmt(m)}")

    if len(grid) > 1:
        x = np.array(grid, dtype=np.float64)
        print("\n=== robustness summary on R1_i2t curve ===")
        print("method\tNoiseAUC(higher better)\tDropSlope(lower better)")
        for name in ["baseline", "ot_only", "prop_only", "gotc"]:
            y = np.array(curve[name], dtype=np.float64)
            auc = float(np.trapezoid(y, x) / max(x[-1] - x[0], 1e-12))
            slope = float((y[0] - y[-1]) / max(x[-1] - x[0], 1e-12))
            print(f"{name}\t{auc:.3f}\t{slope:.3f}")


if __name__ == "__main__":
    main()
