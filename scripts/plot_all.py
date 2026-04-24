#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from plots.plotting import (
    plot_detectability_phase,
    plot_detector_comparison,
    plot_gap_region,
    plot_quality_frontier,
    plot_removability_phase,
    plot_scaling,
)


def main():
    det = pd.read_csv("results/raw/detectability_summary.csv")
    rem = pd.read_csv("results/raw/removability_summary.csv")
    plot_detectability_phase(det, "results/figures/figure1_detectability_llr.png", detector_col="auc_llr")
    plot_detectability_phase(det, "results/figures/figure1_detectability_z.png", detector_col="auc_z")
    plot_removability_phase(rem, "results/figures")
    for source in sorted(det.source.unique()):
        plot_gap_region(det, rem, f"results/figures/gap_region_{source}_greedy.png", source=source, attacker="greedy")
    plot_scaling(det, rem, "results/figures")
    plot_detector_comparison(det, "results/figures/figure5_detector_comparison.png")
    plot_quality_frontier(rem, "results/figures/figure6_quality_removal_frontier.png")


if __name__ == "__main__":
    main()
