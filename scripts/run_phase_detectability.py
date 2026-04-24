#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from experiments.runner import run_detectability_sweep
from utils.io import dump_csv, dump_json, load_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sweeps/detectability.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    df, per_seq = run_detectability_sweep(cfg)
    dump_csv(df, "results/raw/detectability_summary.csv")
    dump_csv(per_seq, "results/raw/detectability_per_sequence.csv")
    dump_json(cfg, "results/logs/detectability_config.json")


if __name__ == "__main__":
    main()
