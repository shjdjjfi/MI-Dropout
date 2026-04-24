#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from plots.plotting import plot_gap_region


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="iid_zipf")
    ap.add_argument("--attacker", default="greedy")
    args = ap.parse_args()
    det = pd.read_csv("results/raw/detectability_summary.csv")
    rem = pd.read_csv("results/raw/removability_summary.csv")
    plot_gap_region(det, rem, f"results/figures/gap_region_{args.source}_{args.attacker}.png", args.source, args.attacker)


if __name__ == "__main__":
    main()
