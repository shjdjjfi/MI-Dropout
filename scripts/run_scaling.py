#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from plots.plotting import plot_scaling


def main():
    det = pd.read_csv("results/raw/detectability_summary.csv")
    rem = pd.read_csv("results/raw/removability_summary.csv")
    plot_scaling(det, rem, "results/figures")


if __name__ == "__main__":
    main()
