#!/usr/bin/env python3
from __future__ import annotations

import pandas as pd


def ci95(x):
    return 1.96 * x.std(ddof=1) / max(len(x) ** 0.5, 1)


def main():
    det = pd.read_csv("results/raw/detectability_summary.csv")
    rem = pd.read_csv("results/raw/removability_summary.csv")

    det_agg = det.groupby(["source", "n", "delta", "gamma"], as_index=False).agg(
        auc_llr_mean=("auc_llr", "mean"),
        auc_llr_ci95=("auc_llr", ci95),
        auc_z_mean=("auc_z", "mean"),
        power_llr_mean=("power_llr", "mean"),
    )
    rem_agg = rem.groupby(["source", "delta", "budget", "epsilon", "attacker"], as_index=False).agg(
        success_mean=("success_rate", "mean"),
        success_ci95=("success_rate", ci95),
        runtime_mean=("avg_runtime", "mean"),
    )
    det_agg.to_csv("results/processed/detectability_agg.csv", index=False)
    rem_agg.to_csv("results/processed/removability_agg.csv", index=False)


if __name__ == "__main__":
    main()
