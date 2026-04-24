from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_detectability_phase(df: pd.DataFrame, out_path: str, detector_col: str = "auc_llr"):
    sources = sorted(df["source"].unique())
    fig, axes = plt.subplots(1, len(sources), figsize=(5 * len(sources), 4), squeeze=False)
    for ax, s in zip(axes[0], sources):
        d = df[df.source == s]
        piv = d.pivot_table(index="delta", columns="n", values=detector_col, aggfunc="mean")
        im = ax.imshow(piv.values, aspect="auto", origin="lower", vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_xticks(np.arange(len(piv.columns)), labels=piv.columns)
        ax.set_yticks(np.arange(len(piv.index)), labels=[f"{x:.2f}" for x in piv.index])
        ax.set_xlabel("n")
        ax.set_ylabel("delta")
        ax.set_title(f"{s} ({detector_col})")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="AUC / Power")
    _save(fig, out_path)


def plot_removability_phase(df: pd.DataFrame, out_dir: str):
    for (source, attacker), d in df.groupby(["source", "attacker"]):
        piv = d.pivot_table(index="delta", columns="budget", values="success_rate", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(piv.values, aspect="auto", origin="lower", vmin=0.0, vmax=1.0, cmap="magma")
        ax.set_xticks(np.arange(len(piv.columns)), labels=[f"{x:.2f}" for x in piv.columns])
        ax.set_yticks(np.arange(len(piv.index)), labels=[f"{x:.2f}" for x in piv.index])
        ax.set_xlabel("budget B")
        ax.set_ylabel("delta")
        ax.set_title(f"{source} - {attacker}")
        fig.colorbar(im, ax=ax, label="Removal success")
        _save(fig, f"{out_dir}/removability_{source}_{attacker}.png")


def plot_gap_region(det_df: pd.DataFrame, rem_df: pd.DataFrame, out_path: str, source: str, attacker: str):
    d = det_df[det_df.source == source].groupby(["n", "delta"], as_index=False)["power_llr"].mean()
    r = rem_df[(rem_df.source == source) & (rem_df.attacker == attacker)].groupby(["delta"], as_index=False)["success_rate"].mean()
    nvals = sorted(d.n.unique()); deltas = sorted(d.delta.unique())
    grid = np.zeros((len(deltas), len(nvals)))
    for i, de in enumerate(deltas):
        rs = float(r[r.delta == de].success_rate.mean()) if (r.delta == de).any() else 0.0
        for j, n in enumerate(nvals):
            p = float(d[(d.delta == de) & (d.n == n)].power_llr.mean())
            if p < 0.8:
                grid[i, j] = 0
            elif rs < 0.5:
                grid[i, j] = 1
            else:
                grid[i, j] = 2
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="Accent", vmin=0, vmax=2)
    ax.set_xticks(np.arange(len(nvals)), labels=nvals)
    ax.set_yticks(np.arange(len(deltas)), labels=[f"{x:.2f}" for x in deltas])
    ax.set_xlabel("n")
    ax.set_ylabel("delta")
    ax.set_title(f"Gap regions: {source}/{attacker}")
    cbar = fig.colorbar(im, ax=ax, ticks=[0,1,2])
    cbar.ax.set_yticklabels(["not detectable", "detectable-not-removable", "detectable-removable"])
    _save(fig, out_path)


def plot_scaling(df_detect: pd.DataFrame, df_remove: pd.DataFrame, out_dir: str):
    fig, ax = plt.subplots(figsize=(5,4))
    sub = df_detect[df_detect.delta > 0]
    s = sub.groupby("n", as_index=False)["type2_llr"].mean().sort_values("n")
    ax.plot(s.n, s.type2_llr, marker="o")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("n")
    ax.set_ylabel("Type II (LLR)")
    ax.set_title("Finite-sample scaling: detection error vs n")
    _save(fig, f"{out_dir}/scaling_detection_vs_n.png")

    fig, ax = plt.subplots(figsize=(5,4))
    s2 = df_remove.groupby("budget", as_index=False)["success_rate"].mean().sort_values("budget")
    ax.plot(s2.budget, s2.success_rate, marker="o")
    ax.set_xlabel("budget B")
    ax.set_ylabel("Removal success")
    ax.set_title("Removal success vs budget")
    _save(fig, f"{out_dir}/scaling_removal_vs_budget.png")


def plot_detector_comparison(df_detect: pd.DataFrame, out_path: str):
    g = df_detect.groupby(["source", "n", "delta"], as_index=False)[["auc_llr", "auc_z"]].mean()
    fig, ax = plt.subplots(figsize=(5,5))
    ax.scatter(g.auc_llr, g.auc_z, alpha=0.6)
    ax.plot([0,1],[0,1],"k--",lw=1)
    ax.set_xlabel("AUC LLR")
    ax.set_ylabel("AUC Green-z")
    ax.set_title("Detector comparison")
    _save(fig, out_path)


def plot_quality_frontier(df_remove: pd.DataFrame, out_path: str):
    g = df_remove.groupby(["attacker", "epsilon"], as_index=False).agg(
        success_rate=("success_rate","mean"),
        quality_loss_proxy=("avg_edit_ratio","mean"),
    )
    fig, ax = plt.subplots(figsize=(5,4))
    for att, d in g.groupby("attacker"):
        ax.plot(d.quality_loss_proxy, d.success_rate, marker="o", label=att)
    ax.set_xlabel("Quality loss proxy (avg edit ratio)")
    ax.set_ylabel("Removal success")
    ax.set_title("Quality-removal frontier")
    ax.legend()
    _save(fig, out_path)
