#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from experiments.runner import run_removability_sweep
from utils.io import dump_csv, dump_json, load_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sweeps/removability.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    df = run_removability_sweep(cfg)
    dump_csv(df, "results/raw/removability_summary.csv")
    dump_json(cfg, "results/logs/removability_config.json")


if __name__ == "__main__":
    main()
